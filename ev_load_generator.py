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


def get_predefined_scenarios():
    """Define the three scenarios from CLAUDE.md."""
    return {
        'high': {
            'name': 'High Load',
            'description': '10 kW per port (14 kW max per pedestal) 8:30 AM - 5:30 PM business days',
            'params': {
                'charging_hours': (8.5, 17.5),
                'energy_range': (70, 90),
                'energy_mean': 80,
                'delay_range': (0.1, 0.5),
                'delay_mean': 0.25,
                'weekdays_only': True
            }
        },
        'medium': {
            'name': 'Medium Load',
            'description': '20-70 kWh per car, 0.25-1 hr between charges (0.66 hr mean)',
            'params': {
                'charging_hours': (8.5, 17.5),
                'energy_range': (20, 70),
                'energy_mean': 45,
                'delay_range': (0.25, 1.0),
                'delay_mean': 0.66,
                'weekdays_only': True
            }
        },
        'low': {
            'name': 'Low Load',
            'description': '15-50 kWh per car, 0.5-2 hr between charges (1.5 hr mean)',
            'params': {
                'charging_hours': (8.5, 17.5),
                'energy_range': (15, 50),
                'energy_mean': 32.5,
                'delay_range': (0.5, 2.0),
                'delay_mean': 1.5,
                'weekdays_only': True
            }
        }
    }


def get_user_input(prompt, input_type=str, default=None, valid_options=None):
    """Get user input with validation and optional default."""
    while True:
        user_input = input(prompt).strip()

        if not user_input and default is not None:
            return default

        if not user_input:
            print("Input required. Please try again.")
            continue

        try:
            value = input_type(user_input)

            if valid_options and value not in valid_options:
                print(f"Invalid option. Choose from: {valid_options}")
                continue

            return value
        except ValueError:
            print(f"Invalid input. Expected {input_type.__name__}")
            continue


def get_custom_scenario_params():
    """Get custom scenario parameters from user."""
    print("\nCustom Scenario Configuration:")
    print("=" * 50)

    params = {}

    print("\nCharging Hours:")
    start_hour = get_user_input("  Start hour (24h format, e.g., 8.5 for 8:30 AM): ", float)
    end_hour = get_user_input("  End hour (24h format, e.g., 17.5 for 5:30 PM): ", float)
    params['charging_hours'] = (start_hour, end_hour)

    print("\nEnergy per Session (kWh):")
    min_energy = get_user_input("  Minimum energy: ", float)
    max_energy = get_user_input("  Maximum energy: ", float)
    mean_energy = get_user_input(f"  Mean energy (default {(min_energy+max_energy)/2:.1f}): ",
                                 float, default=(min_energy+max_energy)/2)
    params['energy_range'] = (min_energy, max_energy)
    params['energy_mean'] = mean_energy

    print("\nDelay Between Sessions (hours):")
    min_delay = get_user_input("  Minimum delay: ", float)
    max_delay = get_user_input("  Maximum delay: ", float)
    mean_delay = get_user_input(f"  Mean delay (default {(min_delay+max_delay)/2:.1f}): ",
                                float, default=(min_delay+max_delay)/2)
    params['delay_range'] = (min_delay, max_delay)
    params['delay_mean'] = mean_delay

    weekdays = get_user_input("\nWeekdays only? (y/n, default y): ", str, default='y',
                              valid_options=['y', 'n', 'yes', 'no'])
    params['weekdays_only'] = weekdays.lower() in ['y', 'yes']

    return params


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("EV Load Profile Generator - SGWS Microgrid Project")
    print("=" * 60)

    # Get charging infrastructure configuration
    port_configs = []

    config_choice = get_user_input("\nUse YAML config file or manual setup? (yaml/manual, default manual): ",
                                   str, default='manual', valid_options=['yaml', 'manual'])

    if config_choice == 'yaml':
        config_file = get_user_input("Enter YAML config file path: ", str)
        try:
            generator = EVLoadProfileGenerator(config_file=config_file)
        except FileNotFoundError:
            print(f"Error: Config file '{config_file}' not found.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    else:
        print("\nDefault Configuration: 6 pedestals with 2 ports each (12 total ports)")
        print("  - Max power per pedestal: 14 kW")
        print("  - Max power per port: 10 kW")

        use_default = get_user_input("Use default configuration? (y/n): ", str,
                                     valid_options=['y', 'n', 'yes', 'no'])

        if use_default.lower() in ['y', 'yes']:
            port_configs = [
                ChargingPortConfig(max_power_kw=14, port_limit_kw=10, num_ports=2)
                for _ in range(6)
            ]
        else:
            num_chargers = get_user_input("Number of charger pedestals: ", int)
            for i in range(num_chargers):
                print(f"\nPedestal {i+1}:")
                max_power = get_user_input("  Max power (kW): ", float)
                port_limit = get_user_input("  Per-port limit (kW): ", float)
                num_ports = get_user_input("  Number of ports: ", int)
                port_configs.append(ChargingPortConfig(max_power, port_limit, num_ports))

        generator = EVLoadProfileGenerator(port_configs=port_configs)

    print(f"\nTotal ports configured: {generator.total_ports}")

    # Get date range
    print("\nDate Range Configuration:")
    print("-" * 30)

    year = get_user_input("Year (default 2024): ", int, default=2024)
    start_month = get_user_input("Start month (1-12): ", int)
    start_day = get_user_input("Start day: ", int)

    end_month = get_user_input("End month (1-12): ", int)
    end_day = get_user_input("End day: ", int)

    start_date = datetime(year, start_month, start_day)
    end_date = datetime(year, end_month, end_day)

    print(f"\nDate range: {start_date.date()} to {end_date.date()}")

    # Get scenario selection
    print("\nScenario Selection:")
    print("-" * 30)

    scenarios = get_predefined_scenarios()

    print("\nAvailable scenarios:")
    print("  1. High Load   - " + scenarios['high']['description'])
    print("  2. Medium Load - " + scenarios['medium']['description'])
    print("  3. Low Load    - " + scenarios['low']['description'])
    print("  4. Custom      - Define your own parameters")

    scenario_choice = get_user_input("\nSelect scenario (1-4): ", int, valid_options=[1, 2, 3, 4])

    if scenario_choice == 1:
        scenario_key = 'high'
        scenario_params = scenarios['high']['params']
        scenario_name = 'high_load'
    elif scenario_choice == 2:
        scenario_key = 'medium'
        scenario_params = scenarios['medium']['params']
        scenario_name = 'medium_load'
    elif scenario_choice == 3:
        scenario_key = 'low'
        scenario_params = scenarios['low']['params']
        scenario_name = 'low_load'
    else:
        scenario_key = 'custom'
        scenario_params = get_custom_scenario_params()
        scenario_name = get_user_input("\nEnter custom scenario name (for filename): ", str)

    # Time resolution
    time_resolution = get_user_input("\nTime resolution in minutes (default 15): ",
                                     int, default=15)

    # Generate load profile
    print("\n" + "=" * 60)
    print("Generating load profile...")

    sessions = generator.generate_charging_sessions(
        start_date=start_date,
        end_date=end_date,
        **scenario_params
    )

    load_profile = generator.sessions_to_load_profile(sessions, time_resolution)

    # Statistics
    print("\nGeneration Complete!")
    print("-" * 30)
    print(f"Scenario: {scenario_name}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Total charging sessions: {len(sessions)}")
    print(f"Time steps: {len(load_profile)}")
    print(f"Peak load: {load_profile['total_load_kw'].max():.1f} kW")
    print(f"Average load: {load_profile['total_load_kw'].mean():.1f} kW")

    if len(sessions) > 0:
        total_energy = sum(s.energy_kwh for s in sessions)
        print(f"Total energy delivered: {total_energy:.1f} kWh")

    # Export
    filename = f"ev_load_{scenario_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
    load_profile.to_csv(filename, index=False)
    print(f"\nLoad profile exported to: {filename}")

    print("\nSample data (first 10 rows):")
    print(load_profile.head(10).to_string())