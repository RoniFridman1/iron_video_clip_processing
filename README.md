### Copy repo and unzip both yolo_tracking.zip and iron_yolo_tracking.zip.

# Running Locally

NOTE: The instructions here are meant to be run on Linux (or WSL). If you are
trying to run directly on MacOS or Windows, good luck :|

*   Poll for new videos in S3 and process them, uploading the results to S3
    under a specific version directory (`--processing-version`). **Any video
    that was previously processed, will not be processed again**

    ```
    python -m ironvideo.clip_processor \
        --save-results \
        --processing-version=v1234
    ```

*   Running the processing on one or more videos (`--video`) from S3, without
    uploading the results to S3 (`--no-save-results`) and keeping the data
    locally for further inspection (`--local-data-path=...`)

    ```
    python -m ironvideo.clip_processor \
        --video=VIDEO-935.mp4 \
        --video=VIDEO-042.mp4 \
        --no-save-results \
        --local-data-path=my-local-dir
    ```

*   Running the processing on specific videos, and uploading the results to S3
    **even if the video previously was already processed**:

    ```
    python -m ironvideo.clip_processor \
        --video=VIDEO-042.mp4 \
        --save-results \
    ```

*   Customizing the S3 path of the input/output and running on all videos in
    these locations:

    ```
    python -m ironvideo.clip_processor \
        --save-results \
        --input-bucket-region-name=us-east-1 \
        --input-video-path=s3://face-watch-aws-cv-prod/or-wilder-source \
        --output-bucket-region-name=il-central-1 \
        --output-video-path=s3://face-shame-proj/output-img
    ```

# Querying the status of a run

*   Query the status of a specific invocation (`--only-status`):

    ```
    python -m ironvideo.clip_processor \
        --only-status \
        --input-bucket-region-name=us-east-1 \
        --input-video-path=s3://face-watch-aws-cv-prod/or-wilder-source \
        --processing-version=v0006
    ```
# Running in the cloud

## Preparation

1.  Create an SSH Key Pair in the EC2 region you plan running VMs on
    *   Here's the [EC2 Key Pair management page](https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#KeyPairs:) in us-east-1
    *   Save the generated key for usage later
2.  Choose/Update a template for the VM from the list of Launch Templates
    *   Here's the [Launch Templates management page](https://us-east-1.console.aws.amazon.com/ec2/home?region=us-east-1#LaunchTemplates:) in us-east-1
    *   **There should be an existing template named `ClipProcessing-Worker`**,
        which has an IAM role assigned, an instance type, an OS and a
        preconfigured disk size
3.  Prepare the installation script - run `python3 -m deployment.make_installer`, which
    will create an installer named `aws_installer.py`
    *   This is a single file that contains all the important files from your
        working directory, and will automatically extra and install them on
        "Amazon Linux" OS (matching the one in the VM template)
4.  **IMPORTANT: By default running in the cloud runs with `--save-results`**,
    so the results will be saved to S3.

## Install the code (without running) on an already created VM

1.  Do the preparation steps above
2.  Run the `start_vms.py` script with the DNS address of the VM to configure,
    and tell the setup script to only extract (`--setup-args='--extract-only'`)

    ```
    python -m deployment.start_vms \
        --vm-addr=ec2-44-211-24-207.compute-1.amazonaws.com \
        --setup-args='--extract-only' \
        --setup-script-path=aws_installer.py \
        --ssh-key-path=~/.ssh/my-ssh-key.pem
    ```

## Start 10 new VMs, Install and Run the Code

1.  Do the preparation steps above
2.  Run the `start_vms.py` script with the VM count, template name (typically
    `ClipProcessing-Worker`) and key pair name (as shown in the AWS console) to
    set the VM with:

    ```
    python -m deployment.start_vms \
        --vm-count=10 \
        --launch-template-name=ClipProcessing-Worker \
        --run-name='My-Funky-Name' \
        --ssh-keypair-name=My-SSH-Key \
        --setup-script-path=aws_installer.py \
        --ssh-key-path=~/.ssh/my-ssh-key.pem  
    ```

    *   The `--run-name` specifies a string that will show up in the AWS EC2
        console so you can identify all VMs from this run
    *   You can pass additional parameters to the actual program via
        `--setup-args` (the setup script will forward most arguments to the
        program), for example `--setup-args='--processing-version=v9876'`
        *   Note that the `--save-results` is controlled directly and not via
            `--setup-args`. By default it will be True.
