import datetime

class DateTimeFormatter:
    """Utility class for generating formatted date and time strings."""
    
    @staticmethod
    def format(option: int) -> str:
        now = datetime.datetime.now()
        today = datetime.date.today()
        
        formats = {
            1: f"Timestamp: {now:%Y-%m-%d %H:%M:%S}",
            2: f"Timestamp: {now:%Y-%b-%d %H:%M:%S}",
            3: f"Date now: {now}",
            4: f"Date today: {today}"
        }
        
        return formats.get(option, str(now))