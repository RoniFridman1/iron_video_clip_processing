import argparse
import logging
import dataclasses
import datetime


@dataclasses.dataclass
class BaseConfig:
    def add_to_argparse(self, parser: argparse.ArgumentParser):
        for field in dataclasses.fields(self):
            field_arg = '--' + field.name.replace('_', '-')
            if field.type is bool:
                parser.add_argument(
                    field_arg, type=field.type, default=field.default,
                    required=False, help=field.metadata.get('help'),
                    # Disable via --no-foo, or enable via --foo
                    action=argparse.BooleanOptionalAction)
            elif field.type in (str, int, float):
                parser.add_argument(
                    field_arg, type=field.type, default=field.default,
                    required=False, help=(
                            field.metadata.get('help') + '\n'
                            + f'(default: {field.default})'))
            else:
                logging.debug('Not adding field %s to argparse (type %s)',
                              field.name, field.type)

    def populate_from_argparse(self, parse_results):
        for field in dataclasses.fields(self):
            if hasattr(parse_results, field.name):
                setattr(self, field.name, getattr(parse_results, field.name))

    def pretty_string(self):
        return '\n'.join(f'{key}: {value}' for key, value in dataclasses.asdict(self).items())


S3_PATH_PREFIX = 's3://'


@dataclasses.dataclass
class S3Path:
    bucket_name: str
    prefix: str

    @classmethod
    def from_path(cls, path: str):
        if not path.startswith(S3_PATH_PREFIX):
            raise ValueError(f'Not a valid s3 path {path} - missing {S3_PATH_PREFIX}')
        name, prefix = path[len(S3_PATH_PREFIX):].split('/', maxsplit=1)
        if not name:
            raise ValueError(f'Empty S3 bucket name in {path}')
        return cls(bucket_name=name, prefix=prefix)

    def to_path(self):
        return f'{S3_PATH_PREFIX}{self.bucket_name}/{self.prefix}'

    def __str__(self) -> str:
        return self.to_path()


@dataclasses.dataclass
class Config(BaseConfig):
    video_bucket: str = dataclasses.field(
        default='face-shame-proj',
        kw_only=True,
        metadata={
            'help': 'The S3 bucket to use for video processing files'
        }
    )
    input_bucket_region_name: str = dataclasses.field(
        default='il-central-1',
        kw_only=True,
        metadata={
            'help': 'The AWS region through which to access the input S3 bucket'
        }
    )
    output_bucket_region_name: str = dataclasses.field(
        default='il-central-1',
        kw_only=True,
        metadata={
            'help': 'The AWS region through which to access the output S3 bucket'
        }
    )
    input_video_path: str = dataclasses.field(
        default='s3://face-shame-proj/source-vid',
        kw_only=True,
        metadata={
            'help': 'The s3://bucket_name/bucket/path from which to read the '
                    'input videos. This should *NOT* have a trailing slash'
        }
    )
    output_video_path: str = dataclasses.field(
        default='s3://face-shame-proj/output-debug',
        kw_only=True,
        metadata={
            'help': 'The s3://bucket_name/bucket/path to which to write the '
                    'output videos. This should *NOT* have a trailing slash'
        }
    )
    dynamodb_table: str = dataclasses.field(
        default='ClipProcessingStatus',
        kw_only=True,
        metadata={
            'help': 'The DynamoDB table used to track the status of video '
                    'processing and coordinate between workers'
        }
    )
    dynamodb_region_name: str = dataclasses.field(
        default='us-east-1',
        kw_only=True,
        metadata={
            'help': 'The AWS region through which to access the DynamoDB table'
        }
    )
    processing_version: str = dataclasses.field(
        default='v0006',
        kw_only=True,
        metadata={
            'help': 'The processing version to mark on this run'
        }
    )
    enable_codeformer_restore: bool = dataclasses.field(
        default=False,
        kw_only=True,
        metadata={
            'help': 'Whether to enable face restoration with CodeFormer'
        }
    )
    save_results: bool = dataclasses.field(
        default=False,
        kw_only=True,
        metadata={
            'help': 'Whether to upload results to AWS (and update in '
                    'S3/DynamoDB/etc.). When `false`, this is a dry run'
        }
    )
    stale_threshold_sec: int = dataclasses.field(
        default=60 * 60 * 2,
        kw_only=True,
        metadata={
            'help': 'Time (in seconds) until a video being processed is '
                    'considered stuck and can be retried'
        }
    )
    enable_postprocess_split: bool = dataclasses.field(
        default=True,
        kw_only=True,
        metadata={
            'help': 'If true, enable splitting tracked people in '
                    'post-processing, using face-clustering algorithms'
        }
    )
    enable_postprocess_merge: bool = dataclasses.field(
        default=True,
        kw_only=True,
        metadata={
            'help': 'If true, enable merging tracked people in '
                    'post-processing, using face-clustering algorithms'
        }
    )
    upload_ignore_dirs: frozenset[str] = dataclasses.field(
        default=frozenset({
            'search_picture',
            'frames_with_bboxes',
            'all_pictures_ratio_margin_50',
            'all_pictures_ratio_margin_100',
            'all_pictures_ratio_margin_200'
        }),
        kw_only=True,
        metadata={
            'help': 'List of directories from the processing results to not '
                    'upload back to S3'
        }
    )

    @property
    def s3_input(self) -> S3Path:
        return S3Path.from_path(self.input_video_path)

    @property
    def s3_output(self) -> S3Path:
        return S3Path.from_path(self.output_video_path)

    @property
    def stale_threshold(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self.stale_threshold_sec)

    @property
    def dry_run(self) -> bool:
        return not self.save_results


@dataclasses.dataclass
class VMConfig(BaseConfig):
    ec2_region_name: str = dataclasses.field(
        default='us-east-1',
        kw_only=True,
        metadata={
            'help': 'The AWS region on which we should start the VMs'
        }
    )
    launch_template_name: str = dataclasses.field(
        default='',
        kw_only=True,
        metadata={
            'help': 'The template to use for launching the VMs'
        }
    )
    ssh_key_path: str = dataclasses.field(
        default='',
        kw_only=True,
        metadata={
            'help': 'The path from which to load the SSH path',
        }
    )
    ssh_keypair_name: str = dataclasses.field(
        default='',
        kw_only=True,
        metadata={
            'help': 'The name of the SSH Key Pair to pick for the VM',
        }
    )
    setup_script_path: str = dataclasses.field(
        default='',
        kw_only=True,
        metadata={
            'help': 'The path/prefix in the bucket to which to upload the '
                    'processing results. This should *NOT* have a trailing slash'
        }
    )
    ssh_binary: str = dataclasses.field(
        default='ssh',
        kw_only=True,
        metadata={
            'help': 'The name (or full path) to the SSH binary'
        }
    )
    scp_binary: str = dataclasses.field(
        default='scp',
        kw_only=True,
        metadata={
            'help': 'The name (or full path) to the SCP binary'
        }
    )
    save_results: bool = dataclasses.field(
        default=True,
        kw_only=True,
        metadata={
            'help': 'Whether to upload results to AWS (and update in '
                    'S3/DynamoDB/etc.). When `false`, this is a dry run'
        }
    )
    setup_args: str = dataclasses.field(
        default='',
        kw_only=True,
        metadata={
            'help': 'Additional args to pass to the setup/run'
        }
    )