import botocore.exceptions
import datetime
import pathlib
import functools
import logging
import os
import abc
import enum
import boto3
import tqdm
import shutil
import subprocess
from typing import Any, Generator, Collection, Mapping
from . import config_module


class VideoStatus(enum.Enum):
    UNKNOWN = 0
    NOT_STARTED = 1
    STARTED = 2
    ERROR = 3
    DONE = 4


class CloudAPI(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def ListInputVideos(self) -> Generator[str, None, None]:
        raise NotImplementedError()

    @abc.abstractmethod
    def GetVideoStatus(self, video_id: str) -> (VideoStatus, datetime.datetime):
        raise NotImplementedError()

    @abc.abstractmethod
    def DeleteVideoStatus(self, video_id: str) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def DownloadInputVideo(self, video_id: str, path: pathlib.Path):
        raise NotImplementedError()

    @abc.abstractmethod
    def MarkVideoStartProcessing(self, video_id: str) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def MarkVideoDoneProcessing(self, video_id: str, metadata: Mapping[str, Any]) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def MarkVideoErrorProcessing(self, video_id: str, message: str) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def UploadProcessingResults(self, video_id: str, local_dir: pathlib.Path,
                                *,
                                ignore_dirs: Collection[str] | None = None) -> None:
        raise NotImplementedError()


def list_objects_v2_with_pagination(s3, *args, **kwargs):
    continuation_token = None
    while True:
        new_kwargs = dict(kwargs)
        if continuation_token:
            new_kwargs['ContinuationToken'] = continuation_token
        result = s3.list_objects_v2(*args, **new_kwargs)
        yield from result['Contents']
        if not result['IsTruncated']:
            break
        continuation_token = result['NextContinuationToken']


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc)


def relative_dirlist(path: pathlib.Path):
    path = path.resolve()
    for subpath in path.rglob('*'):
        if not subpath.is_file():
            continue
        yield subpath, str(subpath.relative_to(path))


def upload_to_s3(*,
                 local_dir: pathlib.Path,
                 s3_path: config_module.S3Path,
                 ignore_dirs: Collection[str] | None = None,
                 region_name: str | None = None,
                 use_cli: bool = True) -> None:
    logging.info('Uploading files from %s to %s',
                 local_dir, s3_path)
    if use_cli and (aws_cli_path := shutil.which('aws')):
        env = dict(os.environ)
        if region_name:
            env['AWS_REGION'] = region_name
        cmd = [
            aws_cli_path, 's3', 'cp', '--recursive',
            str(local_dir), str(s3_path),
        ]
        if ignore_dirs:
            for ignore_dir in ignore_dirs:
                cmd.extend(['--exclude', f'*/{str(ignore_dir)}/*'])
        subprocess.check_call(cmd, env=env,
                              stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL)
        return
    # If we're here we don't have the CLI, thus we need to manually
    # upload. This is typically 10x slower when many files are involved.
    if region_name:
        s3_resource = boto3.resource('s3', region_name=region_name)
    else:
        s3_resource = boto3.resource('s3')
    s3_bucket = s3_resource.Bucket(s3_path.bucket_name)
    local_dir = local_dir.resolve()
    # Save paths to upload as a list to have meaningful progress bar
    file_paths = [
        path for path in local_dir.rglob('*')
        if path.is_file()
    ]
    if ignore_dirs:
        file_paths = [
            path for path in file_paths
            if not any(
                parent_dir in ignore_dirs
                for parent_dir in path.relative_to(local_dir).parent.parts
            )
        ]
    for path in tqdm.tqdm(file_paths):
        rel_path = str(path.relative_to(local_dir))
        with path.open('rb') as f:
            # Use upload and not put, as this is blocking till upload is
            # completed.
            s3_bucket.upload_fileobj(
                Key=f'{s3_path.prefix}/{rel_path}',
                Fileobj=f)


class S3Handler:
    def __init__(self, *, region_name: str, s3_path: config_module.S3Path):
        self.region_name = region_name
        self.s3_path = s3_path
        self.resource = boto3.resource('s3', region_name=region_name)
        self.bucket = self.resource.Bucket(s3_path.bucket_name)

    def get_prefix_path(self, path):
        return f'{self.s3_path.prefix}/{path}'

    def get_s3_path(self, path):
        return f'{self.s3_path}/{path}'


class AwsAPI(CloudAPI):
    def __init__(self, config: config_module.Config | None = None) -> None:
        super().__init__()
        self.config = config or config_module.Config()
        self.s3_input = S3Handler(region_name=config.input_bucket_region_name,
                                  s3_path=config.s3_input)
        self.s3_output = S3Handler(region_name=config.output_bucket_region_name,
                                   s3_path=config.s3_output)
        self.dynamodb_resource = boto3.resource('dynamodb', region_name=config.dynamodb_region_name)
        self.dynamodb_table = self.dynamodb_resource.Table(config.dynamodb_table)
        self.processing_version = config.processing_version
        self.running_on_aws = (os.environ['USER'] == 'ec2-user')

    @classmethod
    def _try_add_to_dict(cls, dest, key: str, value_action):
        try:
            dest[key] = value_action()
        except:
            logging.exception('Failed evaluating %s', key)

    @functools.cached_property
    def _runtime_details(self) -> dict[str, str]:
        result = {
            'local_user': os.environ['USER'],
            'running_on_aws': str(self.running_on_aws),
        }
        if self.running_on_aws:
            from ec2_metadata import ec2_metadata
            self._try_add_to_dict(
                result, 'instance_id', lambda: ec2_metadata.instance_id)
            self._try_add_to_dict(
                result, 'address', lambda: ec2_metadata.public_hostname)
            self._try_add_to_dict(
                result, 'instance_type', lambda: ec2_metadata.instance_type)
            self._try_add_to_dict(
                result, 'region', lambda: ec2_metadata.region)
        return result

    def _get_input_path(self, video_id: str) -> str:
        return self.s3_input.get_prefix_path(video_id)

    def _get_s3_input_path(self, video_id: str) -> str:
        return self.s3_input.get_s3_path(video_id)

    def _get_output_path(self, video_id: str) -> str:
        # For some reason for the output they don't take the file name suffix
        base_video_id, _ = os.path.splitext(video_id)
        result = self.s3_output.get_prefix_path(base_video_id)
        if self.processing_version:
            result += f'/{self.processing_version}'
        return result

    def _get_s3_output_path(self, video_id: str) -> config_module.S3Path:
        return config_module.S3Path(
            bucket_name=self.s3_output.s3_path.bucket_name,
            prefix=self._get_output_path(video_id)
        )

    def _put_processing_status(self, video_id: str, status: VideoStatus, *,
                               force_new: bool = False,
                               extra_data: Mapping[str, Any] | None = None) -> bool:
        item = {
            'video': self._get_s3_input_path(video_id),
            'processing_version': self.processing_version,
            'status': status.name,
            'timestamp_utc_sec': utcnow().isoformat(),
            'runtime': self._runtime_details,
            'output_dir': str(self._get_s3_output_path(video_id)),
        }
        if extra_data:
            for key, value in extra_data.items():
                if key in item:
                    raise ValueError(f'Reserved item key {key}!')
                item[key] = value
        try:
            if force_new:
                try:
                    self.dynamodb_table.put_item(
                        Item=item,
                        # attribute_not_exists evaluates against the item with
                        # the same primary-key + sort-key. Thus verifying any
                        # of these doesn't exist is enough.
                        ConditionExpression='attribute_not_exists(video)')
                except self.dynamodb_resource.meta.client.exceptions.ConditionalCheckFailedException:
                    return False
            else:
                self.dynamodb_table.put_item(Item=item)
        except botocore.exceptions.ClientError:
            logging.exception('Exception in DynamoDB trying to put item %s',
                              str(item))
            return False
        return True

    def ListInputVideos(self) -> Generator[str, None, None]:
        try:
            logging.info('Querying list of input videos')
            prefix = f'{self.s3_input.s3_path.prefix}/'
            for obj in self.s3_input.bucket.objects.filter(Prefix=prefix):
                video_id = obj.key[len(prefix):]
                # Skip possible directories
                if video_id and not video_id.endswith('/'):
                    yield video_id
        except botocore.exceptions.ClientError:
            logging.exception('Failed querying list of input videos')

    def GetVideoStatus(self, video_id: str) -> (VideoStatus, datetime.datetime):
        key = {
            'video': self._get_s3_input_path(video_id),
            'processing_version': self.processing_version,
        }
        try:
            response = self.dynamodb_table.get_item(Key=key)
        except botocore.exceptions.ClientError:
            logging.exception('Unknown exception in DynamoDB')
            return VideoStatus.UNKNOWN, utcnow()

        try:
            match response:
                case {'Item': data, **unused}:
                    match data:
                        case {'status': str(status), 'timestamp_utc_sec': str(timestamp_str), **unused}:
                            status = VideoStatus[status]
                            timestamp = datetime.datetime.fromisoformat(timestamp_str)
                            return status, timestamp
                        case _:
                            return VideoStatus.UNKNOWN, utcnow()
                case _:
                    return VideoStatus.NOT_STARTED, utcnow()
        except ValueError:
            logging.exception('Unknown exception parsing status')
            return VideoStatus.UNKNOWN, utcnow()

    def DeleteVideoStatus(self, video_id: str) -> None:
        key = {
            'video': self._get_s3_input_path(video_id),
            'processing_version': self.processing_version,
        }
        try:
            self.dynamodb_table.delete_item(Key=key)
        except botocore.exceptions.ClientError:
            logging.exception('Unknown exception in DynamoDB')

    def DownloadInputVideo(self, video_id: str, path: pathlib.Path):
        logging.info('Downloading video %s to %s', video_id, path)
        with path.open('w+b') as f:
            self.s3_input.bucket.download_fileobj(
                self._get_input_path(video_id),
                f
            )
        logging.info('Download complete')

    def MarkVideoStartProcessing(self, video_id: str) -> bool:
        logging.info('Marking video %s as started', video_id)
        return self._put_processing_status(video_id, VideoStatus.STARTED, force_new=True)

    def MarkVideoDoneProcessing(self, video_id: str, metadata: Mapping[str, Any]) -> None:
        logging.info('Marking video %s as done', video_id)
        self.s3_output.bucket.put_object(
            Key=self._get_output_path(video_id) + '.done',
            Body='Done'
        )
        self._put_processing_status(video_id, VideoStatus.DONE, extra_data={
            'metadata': metadata
        })

    def MarkVideoErrorProcessing(self, video_id: str, message: str) -> None:
        self._put_processing_status(video_id, VideoStatus.ERROR,
                                    extra_data={
                                        'error_message': message
                                    })

    def UploadProcessingResults(self, video_id: str, local_dir: pathlib.Path,
                                *,
                                ignore_dirs: set[str] | None = None) -> None:
        logging.info('Uploading processing results for video %s', video_id)
        upload_to_s3(
            local_dir=local_dir,
            s3_path=self._get_s3_output_path(video_id),
            region_name=self.config.output_bucket_region_name,
            ignore_dirs=ignore_dirs
        )