import argparse
from collections import defaultdict
import contextlib
from . import config_module
from .script import utils
import pathlib
import time
import tempfile
import traceback

from ironvideo import aws_api
import sys

sys.path.append(
    str((pathlib.Path(__file__).resolve().parent.parent / 'yolo_tracking').resolve())
)

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class MainLogic:
    def __init__(self, config: config_module.Config,
                 *,
                 init_processor: bool = True) -> None:
        self.config = config
        self.api = aws_api.AwsAPI(config)
        if init_processor:
            # Hide heave imports here to have a quick start
            from ironvideo.script import faces_in_video_lib
            self.processor = faces_in_video_lib.ProcessVideo(
                enable_codeformer=config.enable_codeformer_restore)

    def process_video(self, *,
                      video_id: str, run_dir: pathlib.Path,
                      force_process: bool = False):
        logging.info('[START] Starting to process video %s', video_id)
        if not self.config.dry_run:
            should_run = self.api.MarkVideoStartProcessing(video_id)
            if not should_run:
                if force_process:
                    logging.warn('Force run on video taken by another job')
                else:
                    logging.warn('Taken by another job, moving on')
                    return
        out_dir = run_dir / 'output'
        clip_path = run_dir / video_id
        self.api.DownloadInputVideo(video_id, run_dir / video_id)
        logging.info('Running processing')
        metadata = self.processor.naive_process_video(
            video_path=str(clip_path),
            output_dir=str(out_dir),
            output_version=self.config.processing_version,
            enable_postprocess_merge=self.config.enable_postprocess_merge,
            enable_postprocess_split=self.config.enable_postprocess_split
        )
        logging.info('Processing complete')
        if not self.config.dry_run:
            with utils.TimeTracker('s3_upload_duration', metadata):
                self.api.UploadProcessingResults(
                    video_id, out_dir,
                    ignore_dirs=self.config.upload_ignore_dirs)
        if not self.config.dry_run:
            self.api.MarkVideoDoneProcessing(video_id, metadata=metadata)
        logging.info('[FINISH] Done processing video %s with %s', video_id, str(metadata))

    def should_process_video(self, video_id: str) -> bool:
        status, utc_timestamp = self.api.GetVideoStatus(video_id)

        if status == aws_api.VideoStatus.DONE:
            return False

        if status == aws_api.VideoStatus.STARTED:
            if aws_api.utcnow() - utc_timestamp < self.config.stale_threshold:
                # Video actively being processed
                return False
            else:
                logging.warn(f'Resuming likely stuck video "{video_id}" from {utc_timestamp.isoformat()}')
                # Delete so that we can fight on creating the new record
                self.api.DeleteVideoStatus(video_id)
                return True

        if status == aws_api.VideoStatus.NOT_STARTED:
            return True

        if status == aws_api.VideoStatus.ERROR:
            return False

        logging.error(f'Unexpected video status {status} for "{video_id}"')
        return False

    def busy_loop(self):
        while True:
            found = False
            for video_id in self.api.ListInputVideos():
                try:
                    if self.should_process_video(video_id=video_id):
                        found = True
                        with tempfile.TemporaryDirectory() as run_dir_str:
                            self.process_video(
                                video_id=video_id,
                                run_dir=pathlib.Path(run_dir_str))
                except (KeyboardInterrupt, SystemExit):
                    logging.exception('Interrupt requested, exiting')
                    return
                except Exception:
                    logging.exception(
                        'Failed checking/processing video "%s"', video_id)
                    self.api.MarkVideoErrorProcessing(video_id,
                                                      traceback.format_exc())
                    continue
            if not found:
                time.sleep(60)


def main(argv=None):
    config = config_module.Config()
    arg_parser = argparse.ArgumentParser('Poll and process video clips')
    arg_parser.add_argument(
        '--print-config', type=bool, action=argparse.BooleanOptionalAction,
        default=False, help='If set, print the inferred config')
    arg_parser.add_argument(
        '--only-status', type=bool, action=argparse.BooleanOptionalAction,
        default=False, help='If set, print the status of each video')
    arg_parser.add_argument(
        '--init-only', type=bool, action=argparse.BooleanOptionalAction,
        default=False, help='If set, only initialize once and download models')
    arg_parser.add_argument(
        '--video', action='append', dest='videos',
        help='If set, run just these videos and ignore their existing '
             'processing status. No other videos will be processed - the program '
             'will exit when done.')
    arg_parser.add_argument(
        '--local-data-path', type=str, default='',
        help='If running specific videos AND this is set, keep their '
             'processing results in this directory instead of deleting them')
    config.add_to_argparse(arg_parser)
    parsed_args = arg_parser.parse_args(argv)
    config.populate_from_argparse(parsed_args)

    if parsed_args.print_config:
        logging.info('Run config is:\n%s', config.pretty_string())

    main_logic = MainLogic(config, init_processor=not parsed_args.only_status)

    if parsed_args.init_only:
        return

    if parsed_args.only_status:
        status_to_videos = defaultdict(list)
        total_videos = 0
        for video_id in main_logic.api.ListInputVideos():
            status, _ = main_logic.api.GetVideoStatus(video_id)
            status_to_videos[status].append(video_id)
            total_videos += 1

        for status, videos in status_to_videos.items():
            print(f'{status}: {round(len(videos) / total_videos * 100, 2)}%')
            for video in videos:
                print(f'  {video}')
            print()
        return

    if parsed_args.videos:
        if parsed_args.local_data_path:
            local_data_path = pathlib.Path(parsed_args.local_data_path)
            local_data_path.mkdir(parents=False, exist_ok=True)
        else:
            local_data_path = None

        for video_id in parsed_args.videos:
            with contextlib.ExitStack() as exit_stack:
                if local_data_path:
                    video_dir = local_data_path / video_id
                    video_dir.mkdir()
                else:
                    video_dir_str = exit_stack.enter_context(
                        tempfile.TemporaryDirectory()
                    )
                    video_dir = pathlib.Path(video_dir_str)
                main_logic.process_video(video_id=video_id,
                                         run_dir=video_dir,
                                         force_process=True)
        return

    main_logic.busy_loop()


if __name__ == '__main__':
    main()