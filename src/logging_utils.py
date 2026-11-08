
import logging
import sys

class Tags:
    PHASE = "PHASE"
    ATOM_SCOUTING = "ATOM_SCOUTING"
    ATOM_SCOUTING_TRAIN = "ATOM_SCOUTING_TRAIN"
    GLOBAL_OPTIM = "GLOBAL_OPTIM"
    ATOM_PRUNING = "ATOM_PRUNING"
    MODEL = "MODEL"
    DATA = "DATA"
    EXT = "EXT"
    
class TagFilter(logging.Filter):
    """
    A custom filter that only allows records with specific tags 
    to pass through to the handler.
    """
    def __init__(self, allowed_tags=None, allow_untagged=True):
        super().__init__()
        self.allowed_tags = set(allowed_tags) if allowed_tags else set()
        self.allow_untagged = allow_untagged

    def filter(self, record):
        if not hasattr(record, 'tag'):
            record.tag = Tags.EXT

        # Retrieve the 'tag' from the extra dictionary, default to None
        tag = getattr(record, 'tag', None)
        
        # If message has no tag, follow the allow_untagged rule
        if tag is None:
            return self.allow_untagged
            
        # Otherwise, check if the tag is in our allowed list
        return tag in self.allowed_tags
    
def setup_logging(
    log_file="experiment.log", 
    console_tags=None, 
    file_tags=None,
    clear_file=True
):
    """
    Configures the logging system with dual outputs.
    
    Args:
        console_tags (list): List of tags to show in console (e.g. ['PHASE', 'PRUNE']). 
                             If None, defaults to high-level info only.
        file_tags (list): List of tags to log to file. If None, all CDP tags are logged.
        clear_file (bool): Whether to clear the log file at start.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers if re-running in notebooks/scripts
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- Formatters ---
    # Console: Short and readable
    console_format = logging.Formatter('%(message)s')
    # File: Detailed with timestamp and tags
    file_format = logging.Formatter('%(asctime)s | %(levelname)-8s | [%(tag)-7s] | %(message)s')

    # --- 2. File Handler (The "Black Box") ---
    # Usually we want the file to store EVERYTHING for debugging later
    mode = 'w' if clear_file else 'a'
    file_handler = logging.FileHandler(log_file, mode=mode)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)

    if file_tags is None:
        file_tags = [Tags.PHASE, Tags.ATOM_SCOUTING, Tags.GLOBAL_OPTIM, Tags.MODEL, Tags.DATA, Tags.ATOM_PRUNING, Tags.ATOM_SCOUTING_TRAIN]
    
    # If specific file tags are requested, apply filter. Otherwise, log all.
    file_handler.addFilter(TagFilter(allowed_tags=file_tags, allow_untagged=False))
    
    logger.addHandler(file_handler)

    # --- 3. Console Handler (The "Monitor") ---
    # We usually only want to see high-level flow, not every gradient step
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Default to INFO
    console_handler.setFormatter(console_format)
    
    # Apply the Tag Filter
    # Default behavior: Show PHASES and PRUNING, hide inner loop noise
    if console_tags is None:
        console_tags = [Tags.PHASE, Tags.ATOM_SCOUTING, Tags.MODEL, Tags.DATA]
    
    console_handler.addFilter(TagFilter(allowed_tags=console_tags, allow_untagged=True))
    
    logger.addHandler(console_handler)
    
    return logger