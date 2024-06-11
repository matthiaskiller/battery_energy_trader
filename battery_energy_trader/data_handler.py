from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class DataHandler:
    def __init__(self, data_path: str=None, freq: int=60):
        """
        Args:
            data_path (str): Path to the csv data file.
            freq [min] (int): Frequency of the data points (every x minutes). Can be 15 or 60 minutes.
        """
        if data_path is None:
            raise ValueError("Data path must be provided.")
        
        if freq not in [15, 60]:
            raise ValueError("Frequency must be either 15 or 60 [minutes]")
        self.freq = freq

        
        self.data_path = data_path
        self.data = self._load_data()
        self._preprocess_data()
        self._clean_data()

        # How many data points per day are available i.e. 24 if hourly, 96 if each 15min
        self.data_points_per_day = int((24*60)/self.freq)

        # Total days in the dataset
        self.total_days = int(len(self.data) / self.data_points_per_day)
        
    def _load_data(self):
        """
        Load data from csv file into a pandas dataframe.
        """
        return pd.read_csv(self.data_path)
    
    def _preprocess_data(self):
        """
        Preprocess the data by removing irrelevant columns and transforming the time range string into start and end time columns.
        """
        # Remove the last column 'BZN|DE-LU' as entries are empty and it is not relevant for the case study
        self.data = self.data.drop(columns=['BZN|DE-LU'])

        # Retrieve start time from range string
        self.data['Start Time'] = pd.to_datetime(self.data['MTU (CET/CEST)'].apply(lambda x: x.split('-')[0].strip()), format='%d.%m.%Y %H:%M')

        # Retrieve end time from range string
        self.data['End Time'] = pd.to_datetime(self.data['MTU (CET/CEST)'].apply(lambda x: x.split('-')[1].strip()), format='%d.%m.%Y %H:%M')

        # Drop the original time column as it is not needed anymore
        self.data = self.data.drop(columns=['MTU (CET/CEST)'])

        ## Robustness checks on data
        # Make sure the data is sorted by start time
        self.data = self.data.sort_values(by='Start Time')

        # Check if start and end time are set accordingly to the specifications (i.e. 1 hour appart or 15min appart)
        time_diff = self.data['End Time'] - self.data['Start Time']

        # Raise an error if the time difference is not alligned with the data specification
        if not (time_diff == pd.Timedelta(minutes=self.freq)).all():
            raise ValueError("The time difference between start and end time is not 1 hour as specified in the data")
    
        # Add empty columns for optimization results
        self.data['Charge (MW)'] = np.nan
        self.data['Discharge (MW)'] = np.nan
        self.data['SoC (MWh)'] = np.nan
        self.data['Profit per day (EUR)'] = np.nan


    def _clean_data(self):
        """
        Clean the data by interpolating missing price entries and filling missing currency entries.
        """
        # Detect missing price entries in the dataset and linearly interpolate them -> we could also use another strategy here like but I think interpolation makes the most sense
        self.data['Day-ahead Price [EUR/MWh]'] = self.data['Day-ahead Price [EUR/MWh]'].interpolate(method='linear')

        # Detect missing currencies and fall back to EUR as default
        self.data['Currency'] = self.data['Currency'].fillna(value="EUR") 


    ### Plotting functions   

    def plot_price_daily_averaged(self, plot_variance: bool=False):
        """
        Plot the daily averaged prices over time.
        Args:
            plot_variance (bool): Whether to plot the variance of the prices as well.
        """
        self.data['Day'] = self.data['Start Time'].dt.date
        daily_averaged_prices = self.data.groupby('Day')['Day-ahead Price [EUR/MWh]'].mean()
        if plot_variance:
            std= self.data.groupby('Day')['Day-ahead Price [EUR/MWh]'].std()
        plt.figure(figsize=(12, 6))
        daily_averaged_prices.plot(kind='line')
        if plot_variance:
            plt.fill_between(daily_averaged_prices.index, daily_averaged_prices - std, daily_averaged_prices + std, alpha=0.2)
        plt.title('Daily Averaged Day-ahead Prices')
        plt.xlabel('Day')
        plt.ylabel('Price (EUR/MWh)')
        plt.grid(True)
        plt.show()

    def plot_hourly_price_averaged(self, plot_variance: bool=False):
        """
        Plot the hourly prices averaged over time.
        Args:
            plot_variance (bool): Whether to plot the variance of the prices as well.
        """
        # Extract hour of the day from 'Start Time'
        self.data['Hour of Day'] = self.data['Start Time'].dt.hour

        mean_prices = self.data.groupby('Hour of Day')["Day-ahead Price [EUR/MWh]"].mean()
        if plot_variance:
            std= self.data.groupby('Hour of Day')["Day-ahead Price [EUR/MWh]"].std()
        
        plt.figure(figsize=(12, 6))
        if plot_variance:
            plt.errorbar(mean_prices.index, mean_prices, yerr=std, fmt='-o', ecolor='r', capthick=2, alpha=0.7, label='Mean Price +/- Std Dev')
        else:
            mean_prices.plot(kind='line', marker='o', label='Mean Price')
        # mean_prices.plot(kind='line')
        # if plot_variance:
        #     plt.fill_between(mean_prices.index, mean_prices - std, mean_prices + std, alpha=0.2)
        plt.title('Hourly Averaged Day-ahead Prices')
        plt.xlabel('Hour of Day')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(0, 24))
        plt.show()

    def plot_15min_price_averaged(self, plot_variance: bool=False):
        """
        Plot the 15min prices averaged over time.
        Args:
            plot_variance (bool): Whether to plot the variance of the prices as well.
        """
        if self.freq != 15:
            raise ValueError("This function is only available for 15min frequency data")
        
        # Compute an identifier for each 15-minute interval within the day
        self.data['Interval ID'] = self.data['Start Time'].dt.hour * 4 + self.data['Start Time'].dt.minute / 15
        
        # Compute the mean and standard deviation of the prices for each 15min interval across all days
        mean_prices = self.data.groupby('Interval ID')["Day-ahead Price [EUR/MWh]"].mean()
        if plot_variance:
            std = self.data.groupby('Interval ID')["Day-ahead Price [EUR/MWh]"].std()

        plt.figure(figsize=(12, 6))
        if plot_variance:
            plt.errorbar(mean_prices.index, mean_prices, yerr=std, fmt='-o', ecolor='r', capthick=2, alpha=0.7, label='Mean Price +/- Std Dev')
        else:
            mean_prices.plot(kind='line', marker='o', label='Mean Price')
        
        # mean_prices.plot(kind='line')
        # if plot_variance:
        #     plt.fill_between(mean_prices.index, mean_prices - std, mean_prices + std, alpha=0.2)

        plt.title('15-Minute Averaged Day-ahead Prices')
        plt.xlabel('Time of Day')
        plt.ylabel('Price (EUR/MWh)')
        plt.legend()
        plt.grid(True)

        # Adjust x-ticks to only show hourly labels
        hourly_ticks = np.arange(0, 96, 4)  # There are 4 intervals per hour, so we step by 4 to get hourly marks
        tick_labels = [f'{int(i//4)}:00' for i in hourly_ticks]  # Label only the hour, assuming 00 minutes for simplicity

        plt.xticks(hourly_ticks, tick_labels, rotation=45, ha="right")

        plt.tight_layout()
    plt.show()
    
    def plot_daily_price_over_week(self, plot_variance: bool= False):
        """
        Plot the daily prices over a week.
        Args:
            plot_variance (bool): Whether to plot the variance of the prices as well.
        """
        # Group by day and take the first value of each day
        self.data['Weekday'] = self.data['Start Time'].dt.dayofweek
        daily_prices = self.data.groupby('Weekday')['Day-ahead Price [EUR/MWh]'].first()

        if plot_variance:
            std= self.data.groupby('Weekday')['Day-ahead Price [EUR/MWh]'].std()
        plt.figure(figsize=(12, 6))
        daily_prices.plot(kind='line')
        if plot_variance:
            plt.fill_between(daily_prices.index, daily_prices - std, daily_prices + std, alpha=0.2)
        plt.title('Daily Prices over a Week')
        plt.xlabel('Weekday')
        plt.ylabel('Price (EUR/MWh)')
        plt.xticks(range(0, 7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        plt.grid(True)
        plt.show()
        
    def plot_daily_profit(self):
        """
        Plot the daily profit over time.
        """
        # Group by day and take the first value of each day
        self.data['Day'] = self.data['Start Time'].dt.date
        daily_profit = self.data.groupby('Day')['Profit per day (EUR)'].first()  # Since profit is repeated, take the first value of each day

        plt.figure(figsize=(12, 6))
        daily_profit.plot(kind='bar')
        plt.title('Daily Profit from Battery Operations')
        plt.xlabel('Day')
        plt.ylabel('Profit (EUR)')

        # Generate a list of labels with every 10th day, leaving others as empty strings
        tick_labels = [label if idx % 10 == 0 else '' for idx, label in enumerate(daily_profit.index)]
        plt.xticks(range(len(daily_profit)), tick_labels, rotation=45)  # Set custom tick labels

        plt.tight_layout()  # Adjust layout to make room for the x-axis labels
        plt.show()

    def plot_averaged_charge_discharge_over_day(self):
        """
        Plot charging and discharging rates by hour of day averaged over all days. 
        """
        # Extract hour of the day from 'Start Time'
        self.data['Hour of Day'] = self.data['Start Time'].dt.hour

        # Calculate mean charging and discharging rates by hour of day
        mean_rates = self.data.groupby('Hour of Day')[['Charge (MW)', 'Discharge (MW)']].mean()

        plt.figure(figsize=(10, 6))
        mean_rates['Charge (MW)'].plot(label='Average Charging Rate', marker='^')
        mean_rates['Discharge (MW)'].plot(label='Average Discharging Rate', marker='v')
        plt.title('Average Charging and Discharging Rates by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Power (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(0, 24))
        plt.show()

    def plot_soc(self, day: int=0):
        """
        Plot the state of charge (SoC) over time for a given day.
        Args:
            day (int): The day to be plotted.
        """

        sample_day = self.data['Day'].unique()[day] # set the day to be plotted
        sample_df = self.data[self.data['Day'] == sample_day]

        plt.figure(figsize=(12, 6))
        plt.plot(sample_df['Start Time'], sample_df['SoC (MWh)'], marker='o', linestyle='-', label='SoC (MWh)')

        plt.title(f'State of Charge (SoC) Throughout Day {day}: {sample_day}')
        plt.xlabel('Time')
        plt.ylabel('State of Charge (MWh)')
        plt.grid(True)

        # Format x-axis to show hour of day
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)

        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_charge_discharge_over_day(self, day: int=0):
        """
        Plot charging and discharging rates by hour of day on a given day. 
        """
        # Extract hour of the day from 'Start Time'
        self.data['Hour of Day'] = self.data['Start Time'].dt.hour

        # reset index
        data_day = self.data[day*24:day*24+24].reset_index(drop=True)

        plt.figure(figsize=(10, 6))
        data_day['Charge (MW)'].plot(label='Charging Rate', marker='^')
        data_day['Discharge (MW)'].plot(label='Discharging Rate', marker='v')
        plt.title(f'Charging and Discharging Rates on Day {day}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Power (MW)')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(0, 24))
        plt.show()