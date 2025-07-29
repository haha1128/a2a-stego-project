from datetime import datetime
import config
class TimestampMannager:
    def __init__(self):
        pass
    @staticmethod
    def get_valid_timestamp(function)->float|None:
        """
        Args:
            function: The condition that the timestamp needs to satisfy.
        Returns:
            A timestamp (float) or None.
        """
        timestamp= None
        i = 0
        while i < config.TIMESTAMP_MAX_TRY:
            timestamp = datetime.now().timestamp()
            if function(timestamp):
                return timestamp
            i += 1
        return timestamp
        
    @staticmethod
    def is_valid_timestamp(timestamp:float,function)->bool:
        """
        Checks if a timestamp satisfies a condition.
        Args:
            timestamp: The timestamp to check.
            function: The condition that the timestamp needs to satisfy.
        Returns:
            Whether the condition is satisfied (bool).
        """
        return function(timestamp)