import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logging.info(f'Processing event: {event!r}')
    match event:
        case {'vid_id': str(vid_id)}:
            pass
        case _:
            logging.error(f'Invalid event!')
            return
    # TODO: Logic here
