import argparse
import boto3
import logging
from ironvideo import config_module
import pathlib
import os
import subprocess
import shutil
import shlex
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def resolve_file_path(path_str: str) -> pathlib.Path:
    path = pathlib.Path(os.path.expanduser(path_str)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'File not found at {path_str}')
    return path


def resolve_binary_path(name_or_path: str) -> pathlib.Path:
    if os.sep in name_or_path:
        path = pathlib.Path(os.path.expanduser(path_str))
        if not path.is_file():
            raise FileNotFoundError(f'Binary not found at {name_or_path}')
    else:
        path_str = shutil.which(name_or_path)
        if not path_str:
            raise ValueError(f'Binary {name_or_path} not found on the path')
        path = pathlib.Path(path_str)
    return path.resolve()


class VMLaunch:
    def __init__(self,
                 config: config_module.VMConfig) -> None:
        self.ec2 = boto3.resource('ec2', region_name=config.ec2_region_name)
        self.ssh_key_path = resolve_file_path(config.ssh_key_path)
        self.setup_script_path = resolve_file_path(config.setup_script_path)
        self.ssh_binary = resolve_binary_path(config.ssh_binary)
        self.scp_binary = resolve_binary_path(config.scp_binary)
        self.launch_template_name = config.launch_template_name
        self.key_name = config.ssh_keypair_name
        self.save_results = config.save_results
        self.setup_args = config.setup_args

    def launch_instances(self, *,
                         count: int,
                         run_name: str):
        logging.info('Sending request to launch %s VM instances', count)
        instances = self.ec2.create_instances(
            LaunchTemplate={'LaunchTemplateName': self.launch_template_name},
            KeyName=self.key_name,
            MinCount=count, MaxCount=count)
        logging.info('Request succeeded, waiting for VMs to come up')
        
        for i, instance in enumerate(instances):
            instance.wait_until_running()
            instance.reload()
            # Warning: This can only be done when the instance was already
            # created (and the above request is async)
            name = '%s (%04d/%04d, %s, %s)' % (
                run_name,
                i + 1, count, self.key_name, self.launch_template_name
            )
            instance.create_tags(Tags=[{
                'Key': 'Name',
                'Value': name
            }])
            logging.info('VM %r (%s) is ready', name, instance.public_dns_name)
        
        return instances

    def log_and_check_subprocess_call(self, cmd_args: list[str], **kwargs):
        cmd_str = shlex.join(cmd_args)
        logging.info('[SHELL] %s', cmd_str)
        return subprocess.check_call(cmd_args, **kwargs)

    def start_code_on_instance(self, instance_addr: str):
        logging.info('[INFO] Waiting for VM to answer SSH %s', instance_addr)
        for i in range(10):
            try:
                self.log_and_check_subprocess_call([
                    str(self.ssh_binary), '-i', str(self.ssh_key_path),
                    '-o', 'StrictHostKeychecking=no',
                    f'ec2-user@{instance_addr}',
                    'echo OK'
                ])
                break
            except subprocess.CalledProcessError:
                logging.info('[INFO] SSH not yet ready at %s, sleeping', instance_addr)
                time.sleep(3)
        logging.info('[PREPARE] Copying setup script to %s', instance_addr)
        self.log_and_check_subprocess_call([
            str(self.scp_binary), '-i', str(self.ssh_key_path),
            '-o', 'StrictHostKeychecking=no',
            str(self.setup_script_path),
            f'ec2-user@{instance_addr}:~/aws_installer.py'
        ])
        self.log_and_check_subprocess_call([
            str(self.ssh_binary), '-i', str(self.ssh_key_path),
            '-o', 'StrictHostKeychecking=no',
            f'ec2-user@{instance_addr}',
            'chmod +x ~/aws_installer.py'
        ])
        logging.info('[RUN] Running setup script on %s', instance_addr)
        self.log_and_check_subprocess_call([
            str(self.ssh_binary), '-i', str(self.ssh_key_path),
            '-o', 'StrictHostKeychecking=no',
            f'ec2-user@{instance_addr}',
            (
                'screen -d -m  -L -Logfile execution.log ~/aws_installer.py '
                + '--' + ('' if self.save_results else 'no-') + 'save-results '
                + self.setup_args
            )
        ])


def main(argv=None):
    config = config_module.VMConfig()
    arg_parser = argparse.ArgumentParser('Launch EC2 VM Instances with our code')
    arg_parser.add_argument(
        '--print-config', type=bool, action=argparse.BooleanOptionalAction,
        default=False, help='If set, print the inferred config')
    arg_parser.add_argument(
        '--init-only', type=bool, action=argparse.BooleanOptionalAction,
        default=False, help='If set, only initialize once to verify config')
    arg_parser.add_argument(
        '--vm-count', type=int, default=1,
        help='The number of VMs to launch (1 by default)')
    arg_parser.add_argument(
        '--terminate-failed-setup', type=bool, action=argparse.BooleanOptionalAction,
        default=True, help='If true (default), terminate VMs that failed to '
        'set-up. Only valid when starting new VM instances.')
    arg_parser.add_argument(
        '--run-name', type=str, required=True,
        help='The name of this run (used to identify VMs)')
    config.add_to_argparse(arg_parser)
    arg_parser.add_argument(
        '--vm-addr', action='append', dest='vms',
        help='If set, just configure these already running VMs.')
    parsed_args = arg_parser.parse_args(argv)
    config.populate_from_argparse(parsed_args)

    ec2_homepage = (
        f'https://{config.ec2_region_name}.console.aws.amazon.com'
        f'/ec2/home?region={config.ec2_region_name}'
    )
    if parsed_args.print_config:
        logging.info('Run config is:\n%s', config.pretty_string())

    # We only start new VMs if we are not configuring existing ones
    start_new_vms = not parsed_args.vms

    if start_new_vms:
        if not config.launch_template_name:
            arg_parser.error(
                'Missing `--launch-template-name` - please specify an SSH key!'
                f'\nSee {ec2_homepage}#LaunchTemplates:')

        if not config.ssh_keypair_name:
            arg_parser.error(
                'Missing `--ssh-keypair-name` - please specify an SSH key\n'
                f'See {ec2_homepage}#KeyPairs:')
        
        if not parsed_args.run_name:
            arg_parser.error(
                'Missing `--run-name` (a name to mark VMs of this run)'
            )

    if not config.ssh_key_path:
        arg_parser.error(
            'Missing `--ssh-key-path`! Please specify the file path to your '
            'private SSH key (matching the AWS key). Typically this is a file '
            ' under your `~/.ssh` directory')

    launcher = VMLaunch(config)
    
    if parsed_args.init_only:
        return
    
    if parsed_args.vms:
        for vm_addr in parsed_args.vms:
            try:
                launcher.start_code_on_instance(vm_addr)
            except subprocess.CalledProcessError:
                logging.error('ERROR: Failed setting-up %s', vm_addr)
        return
    
    vms = launcher.launch_instances(
        count=parsed_args.vm_count,
        run_name=parsed_args.run_name
    )
    for vm in vms:
        if not vm.public_dns_name:
            logging.error(
                "ERROR: Can't connect to VM %s - could not resolve it's DNS "
                "from the AWS console!", vm.public_dns_name)
            continue
        try:
            launcher.start_code_on_instance(vm.public_dns_name)
        except subprocess.CalledProcessError:
            logging.error('ERROR: Failed setting-up %s', vm.public_dns_name)
            if parsed_args.terminate_failed_setup:
                logging.error('Terminating %s to avoid compute waste',
                              vm.public_dns_name)
                try:
                    vm.terminate()
                except (KeyboardInterrupt, SystemExit):
                    logging.exception(
                        'ERROR: Failed terminating non-responsive VM %s',
                        vm.public_dns_name)


if __name__ == '__main__':
    main()