from datetime import datetime
from typing import Optional, Dict, Any, Union
from abc import ABC, abstractmethod

class IDateTimeFormatter(ABC):
    """Interface for date and time formatting.
    
    Follows the Interface Segregation Principle by defining a specific formatting interface.
    """
    
    @abstractmethod
    def format(self, format_option: str = 'default') -> str:
        """Format the current date and time.
        
        Args:
            format_option: Option for formatting
            
        Returns:
            Formatted date and time string
        """
        pass

class DateTimeFormatter(IDateTimeFormatter):
    """Handles formatting of date and time.
    
    Implements the IDateTimeFormatter interface and follows the Single Responsibility Principle
    by focusing only on date and time formatting.
    """
    
    # Format options dictionary
    _FORMAT_OPTIONS = {
        'default': '%Y-%m-%d %H:%M:%S',
        'date_only': '%Y-%m-%d',
        'time_only': '%H:%M:%S',
        'filename': '%Y%m%d_%H%M%S',
        'log': '%Y-%m-%d %H:%M:%S',
        'display': '%B %d, %Y at %I:%M %p'
    }
    
    @classmethod
    def format(cls, option: str = 'default', dt: Optional[datetime] = None) -> str:
        """Format the date and time according to the specified option.
        
        Args:
            option: Format option, one of 'default', 'date_only', 'time_only', 'filename', 'log', 'display'
            dt: Datetime object to format, if None, current datetime is used
            
        Returns:
            Formatted date and time string
            
        Raises:
            ValueError: If the option is not valid
        """
        if option not in cls._FORMAT_OPTIONS:
            valid_options = ', '.join(cls._FORMAT_OPTIONS.keys())
            raise ValueError(f"Invalid format option: {option}. Valid options are: {valid_options}")
        
        dt = dt or datetime.now()
        return dt.strftime(cls._FORMAT_OPTIONS[option])
    
    @classmethod
    def get_available_formats(cls) -> Dict[str, str]:
        """Get all available format options.
        
        Returns:
            Dictionary of format options and their format strings
        """
        return cls._FORMAT_OPTIONS.copy()
    
    @classmethod
    def add_format_option(cls, name: str, format_string: str) -> None:
        """Add a new format option.
        
        Args:
            name: Name of the format option
            format_string: Format string for strftime
            
        Raises:
            ValueError: If the name already exists
        """
        if name in cls._FORMAT_OPTIONS:
            raise ValueError(f"Format option '{name}' already exists")
        
        cls._FORMAT_OPTIONS[name] = format_string

class TimestampConverter:
    """Handles conversion between timestamps and datetime objects.
    
    Follows the Single Responsibility Principle by focusing only on timestamp conversion.
    """
    
    @staticmethod
    def to_datetime(timestamp: Union[int, float]) -> datetime:
        """Convert a timestamp to a datetime object.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Datetime object
        """
        return datetime.fromtimestamp(timestamp)
    
    @staticmethod
    def from_datetime(dt: datetime) -> float:
        """Convert a datetime object to a timestamp.
        
        Args:
            dt: Datetime object
            
        Returns:
            Unix timestamp
        """
        return dt.timestamp()