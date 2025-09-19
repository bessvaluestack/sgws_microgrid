import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import random


@dataclass
class ChargingPortConfig:
    """Configuration for a charging port or charger."""
    max_power_kw: float
    port_limit_kw: float
    num_ports: int


@dataclass
class ChargingSession:
    """Represents a single charging session."""
    start_time: datetime
    end_time: datetime
    port_id: int
    energy_kwh: float
    power_kw: float


class EVLoadProfileGenerator:
    """Generator for EV charging load profiles for employee charging scenarios."""

    def __init__(self, config_file: str = None, port_configs: List[ChargingPortConfig] = None):
        """
        Initialize the load profile generator.

        Args:
            config_file: Path to YAML configuration file
            port_configs: List of charging port configurations (alternative to config_file)
        """
        if config_file:
            self.config = self._load_config(config_file)
            self.port_configs = self._parse_port_configs(self.config['charging_infrastructure'])
        elif port_configs:
            self.port_configs = port_configs
            self.config = {}
        else:
            raise ValueError("Either config_file or port_configs must be provided")

        self.total_ports = sum(config.num_ports for config in self.port_configs)

    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _parse_port_configs(self, infrastructure_config: List[Dict]) -> List[ChargingPortConfig]:
        """Parse port configurations from YAML config."""
        configs = []
        for charger in infrastructure_config:
            configs.append(ChargingPortConfig(
                max_power_kw=charger['max_power_kw'],
                port_limit_kw=charger['port_limit_kw'],
                num_ports=charger['num_ports']
            ))
        return configs

    def generate_charging_sessions(
        self,
        start_date: datetime,
        end_date: datetime,
        charging_hours: Tuple[int, int] = (8.5, 17.5),  # 8:30 AM to 5:30 PM
        energy_range: Tuple[float, float] = (20, 70),
        energy_mean: float = 45,
        delay_range: Tuple[float, float] = (0.25, 1.0),
        delay_mean: float = 0.66,
        weekdays_only: bool = True
    ) -> List[ChargingSession]:
        """
        Generate charging sessions for the specified period.

        Args:
            start_date: Start date for generation
            end_date: End date for generation
            charging_hours: Tuple of (start_hour, end_hour) in 24h format
            energy_range: Tuple of (min_energy, max_energy) in kWh
            energy_mean: Mean energy per session in kWh
            delay_range: Tuple of (min_delay, max_delay) between sessions in hours
            delay_mean: Mean delay between sessions in hours
            weekdays_only: If True, only generate sessions on weekdays

        Returns:
            List of ChargingSession objects
        """
        sessions = []
        current_date = start_date

        while current_date <= end_date:
            # Skip weekends if weekdays_only is True
            if weekdays_only and current_date.weekday() >= 5:
                current_date += timedelta(days=1)
                continue

            # Generate sessions for this day
            day_sessions = self._generate_daily_sessions(
                current_date, charging_hours, energy_range, energy_mean,
                delay_range, delay_mean
            )
            sessions.extend(day_sessions)

            current_date += timedelta(days=1)

        return sessions

    def _generate_daily_sessions(
        self,
        date: datetime,
        charging_hours: Tuple[int, int],
        energy_range: Tuple[float, float],
        energy_mean: float,
        delay_range: Tuple[float, float],
        delay_mean: float
    ) -> List[ChargingSession]:
        """Generate charging sessions for a single day."""
        sessions = []

        # Convert hours to datetime objects
        start_hour, end_hour = charging_hours
        day_start = date.replace(hour=int(start_hour), minute=int((start_hour % 1) * 60), second=0, microsecond=0)
        day_end = date.replace(hour=int(end_hour), minute=int((end_hour % 1) * 60), second=0, microsecond=0)

        # Track port availability
        port_availability = {}
        port_id = 0
        for config in self.port_configs:
            for _ in range(config.num_ports):
                port_availability[port_id] = day_start
                port_id += 1

        # Generate sessions throughout the day
        current_time = day_start

        while current_time < day_end:
            # Find available port
            available_ports = [pid for pid, avail_time in port_availability.items()
                             if avail_time <= current_time]

            if not available_ports:
                # Move to next availability time
                next_avail_time = min(port_availability.values())
                current_time = max(current_time + timedelta(minutes=15), next_avail_time)
                continue

            # Select random port
            port_id = random.choice(available_ports)

            # Get port configuration
            port_config = self._get_port_config(port_id)

            # Generate energy for this session using truncated normal distribution
            energy_kwh = self._generate_truncated_normal(
                energy_mean,
                (energy_range[1] - energy_range[0]) / 4,  # std dev
                energy_range[0],
                energy_range[1]
            )

            # Calculate charging power and duration
            power_kw = min(port_config.port_limit_kw, port_config.max_power_kw)
            duration_hours = energy_kwh / power_kw

            # Create session
            session_start = current_time
            session_end = session_start + timedelta(hours=duration_hours)

            # Don't exceed day end
            if session_end > day_end:
                session_end = day_end
                energy_kwh = power_kw * ((session_end - session_start).total_seconds() / 3600)

            session = ChargingSession(
                start_time=session_start,
                end_time=session_end,
                port_id=port_id,
                energy_kwh=energy_kwh,
                power_kw=power_kw
            )
            sessions.append(session)

            # Update port availability
            port_availability[port_id] = session_end

            # Generate delay until next session
            delay_hours = self._generate_truncated_normal(
                delay_mean,
                (delay_range[1] - delay_range[0]) / 4,  # std dev
                delay_range[0],
                delay_range[1]
            )

            current_time += timedelta(hours=delay_hours)

        return sessions

    def _get_port_config(self, port_id: int) -> ChargingPortConfig:
        """Get the configuration for a specific port ID."""
        current_port = 0
        for config in self.port_configs:
            if current_port <= port_id < current_port + config.num_ports:
                return config
            current_port += config.num_ports
        raise ValueError(f"Port ID {port_id} not found")

    def _generate_truncated_normal(self, mean: float, std: float, min_val: float, max_val: float) -> float:
        """Generate a value from truncated normal distribution."""
        while True:
            value = np.random.normal(mean, std)
            if min_val <= value <= max_val:
                return value

    def sessions_to_load_profile(
        self,
        sessions: List[ChargingSession],
        time_resolution_minutes: int = 15
    ) -> pd.DataFrame:
        """
        Convert charging sessions to a time-series load profile.

        Args:
            sessions: List of charging sessions
            time_resolution_minutes: Time resolution for the profile in minutes

        Returns:
            DataFrame with timestamp and total_load_kw columns
        """
        if not sessions:
            return pd.DataFrame(columns=['timestamp', 'total_load_kw'])

        # Find overall time range
        start_time = min(session.start_time for session in sessions)
        end_time = max(session.end_time for session in sessions)

        # Create time index
        time_index = pd.date_range(
            start=start_time.replace(second=0, microsecond=0),
            end=end_time.replace(second=0, microsecond=0),
            freq=f'{time_resolution_minutes}min'
        )

        # Initialize load profile
        load_profile = pd.DataFrame({
            'timestamp': time_index,
            'total_load_kw': 0.0
        })

        # Add each session to the profile
        for session in sessions:
            # Find time slots that overlap with the session
            session_mask = (
                (load_profile['timestamp'] >= session.start_time) &
                (load_profile['timestamp'] < session.end_time)
            )
            load_profile.loc[session_mask, 'total_load_kw'] += session.power_kw

        return load_profile

    def generate_load_profile(
        self,
        start_date: datetime,
        end_date: datetime,
        scenario: str = None,
        charging_hours: Tuple[int, int] = None,
        energy_range: Tuple[float, float] = None,
        energy_mean: float = None,
        delay_range: Tuple[float, float] = None,
        delay_mean: float = None,
        weekdays_only: bool = None,
        time_resolution_minutes: int = None
    ) -> pd.DataFrame:
        """
        Generate a complete load profile for the specified parameters.

        Args:
            scenario: Scenario name from config file (overrides individual parameters)
            Other parameters: Override config file values if provided

        Returns:
            DataFrame with timestamp and total_load_kw columns
        """
        # Use config file values if available and parameters not explicitly provided
        if scenario and 'scenarios' in self.config:
            scenario_config = self.config['scenarios'][scenario]
            charging_hours = charging_hours or tuple(scenario_config['charging_hours'])
            energy_range = energy_range or tuple(scenario_config['energy_per_session']['range'])
            energy_mean = energy_mean or scenario_config['energy_per_session']['mean']
            delay_range = delay_range or tuple(scenario_config['delay_between_sessions']['range'])
            delay_mean = delay_mean or scenario_config['delay_between_sessions']['mean']
            weekdays_only = weekdays_only if weekdays_only is not None else scenario_config.get('weekdays_only', True)

        # Use defaults if still None
        charging_hours = charging_hours or (8.5, 17.5)
        energy_range = energy_range or (20, 70)
        energy_mean = energy_mean or 45
        delay_range = delay_range or (0.25, 1.0)
        delay_mean = delay_mean or 0.66
        weekdays_only = weekdays_only if weekdays_only is not None else True
        time_resolution_minutes = time_resolution_minutes or 15

        sessions = self.generate_charging_sessions(
            start_date, end_date, charging_hours, energy_range, energy_mean,
            delay_range, delay_mean, weekdays_only
        )

        return self.sessions_to_load_profile(sessions, time_resolution_minutes)


if __name__ == "__main__":
    # Example usage with YAML config
    import sys

    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        generator = EVLoadProfileGenerator(config_file=config_file)

        # Generate load profile for one week
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 7)

        # Use medium scenario from config
        load_profile = generator.generate_load_profile(
            start_date=start_date,
            end_date=end_date,
            scenario='medium'
        )

        print(f"Generated load profile with {len(load_profile)} time steps")
        print(f"Peak load: {load_profile['total_load_kw'].max():.1f} kW")
        print(f"Average load: {load_profile['total_load_kw'].mean():.1f} kW")
        print("\nSample data:")
        print(load_profile.head(10))
    else:
        print("Usage: python ev_load_generator.py <config_file.yaml>")
        print("Example config file needed with charging infrastructure and scenarios.")