import cvxpy as cp
import numpy as np

from battery_energy_trader.data_handler import DataHandler

class Optimizer:
    """
    This class is used to optimize the profit of the grid battery.
    """
    def __init__(self, data_handler: DataHandler=None, battery_capacity: float=1, max_power: float=1, efficiency: float=1):
        """
        Args:
            data (DataHandler): DataHandler object containing the data.
            battery_capacity (float): The capacity of the battery in MWh.
            max_power (float): The maximum power of the battery in MW.
            efficiency (float): The efficiency of the battery.  Assuming 100% for now 
        """
        self.data_handler = data_handler
        self.battery_capacity = battery_capacity
        self.max_power = max_power
        self.efficiency = efficiency


    def optimize_lp(self, time_horizon: int=1, solver=cp.ECOS, constraints_soc_end=True):
        """
        Optimize the revenue of the grid battery using a LP solver.
        # TODO: refactor together with MILP method to avoid code duplication
        Args:
            time_horizon (int): The optimization horizon in days.
            solver (cvxpy solver): The solver to use for the optimization. Defaults to ECOS.
            constrain_soc_end (bool): Whether to constrain the SOC at the end of the day to be the same as at the beginning.
        """
        # Set initial SOC to 0
        previous_end_soc = 0
        
        # Number of optimization windows based on the total days and the chosen time horizon
        num_windows = (self.data_handler.total_days + time_horizon - 1) // time_horizon

        for window in range(num_windows):
            # Calculate the start and end index for the current window
            start_day = window * time_horizon
            end_day = min((window + 1) * time_horizon, self.data_handler.total_days)
            start_idx = start_day * self.data_handler.data_points_per_day
            end_idx = end_day * self.data_handler.data_points_per_day
            
            # Adjust daily_prices to span the entire time horizon
            daily_prices = self.data_handler.data['Day-ahead Price [EUR/MWh]'].values[start_idx:end_idx]
            
            # Adjust the definition of charge, discharge, and soc variables
            # to cover the entire time horizon
            charge = cp.Variable((end_day - start_day) * self.data_handler.data_points_per_day)
            discharge = cp.Variable((end_day - start_day) * self.data_handler.data_points_per_day)
            soc = cp.Variable(((end_day - start_day)) * self.data_handler.data_points_per_day) #+1
            

            # Objective -> Maximize profits from buying (charging) low and selling (discharging) high
            objective = cp.Maximize(cp.sum(cp.multiply(discharge - charge, daily_prices)))
            
            start_soc = 0 if constraints_soc_end else previous_end_soc

            # Constraints
            constraints = [
                soc[0] == start_soc,  # Initial SOC is either 0 or the previous end SOC
                soc >= 0, # SOC cannot be negative
                soc <= self.battery_capacity, # SOC cannot exceed battery capacity
                charge >= 0, # Cannot charge negative amounts
                discharge >= 0, # Cannot discharge negative amounts
                charge <= self.max_power, # Charge rate cannot exceed max power
                discharge <= self.max_power, # Discharge rate cannot exceed max power
            ]

            # SOC/Battery dynamics
            for t in range((end_day - start_day) * self.data_handler.data_points_per_day):
                constraints.append(soc[t] == soc[t-1] + (charge[t] * self.efficiency) - (discharge[t] / self.efficiency))

            # Assuming the battery SOC at the end of the day should return to its initial state 
            # Assuming this to ensure continuity between days 
            # -> we can obviously change this to other assumptions or always start with the last value of the previous day
            # This is a strong assumption which influences the outcome significatnly 
            # -> no constraint on last soc value performs way better
            if constraints_soc_end:
                constraints.append(soc[-1] == soc[0])

            # Solve the problem
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=solver)

            # Raise error if the problem is infeasible or unbounded
            if prob.status != "optimal":
                raise ValueError(f"Window {window} opt. propblem is {prob.status}")
            
            # After solving, directly store results in df
            self.data_handler.data.loc[start_idx:end_idx-1, 'Charge (MW)'] = charge.value
            self.data_handler.data.loc[start_idx:end_idx-1, 'Discharge (MW)'] = discharge.value
            self.data_handler.data.loc[start_idx:end_idx-1, 'SoC (MWh)'] = soc.value #[:-1] # Adjust if storing final SoC differently
            self.data_handler.data.loc[start_idx:end_idx-1, 'Profit per day (EUR)'] = result

            previous_end_soc = soc.value[-1]

        # When we use a problem formulation like the above we cannot specify 
        # that we cannot simultaneously charge and discharge.
        # Hence, when the optimal action is to hold the soc and neither charge nor discharge, 
        # the current solver will display this as charge = 0.5 MW and discharge = 0.5MW. 
        # -> Resulting in no flow essentially holding the soc.
        # To enforce a behavior with both values beeing zero if the SOC should be holded we can formulize the problem as an MILP.
        # MILP can use binary varibales like is_charging to constraint this behavior.
            
        # --> For now we just use a post processing step to mitigate this issue

        # Threshold for considering values as zero
        threshold = 1e-6

        # Set small values to zero
        self.data_handler.data['Charge (MW)'] = self.data_handler.data['Charge (MW)'].apply(lambda x: 0 if np.abs(x) < threshold else x)
        self.data_handler.data['Discharge (MW)'] = self.data_handler.data['Discharge (MW)'].apply(lambda x: 0 if np.abs(x) < threshold else x)

        # Identify rows where both charging and discharging are non-zero (above the threshold)
        simultaneous_indices = (self.data_handler.data['Charge (MW)'] > threshold) & (self.data_handler.data['Discharge (MW)'] > threshold)

        # Set both to zero for those rows
        self.data_handler.data.loc[simultaneous_indices, 'Charge (MW)'] = 0
        self.data_handler.data.loc[simultaneous_indices, 'Discharge (MW)'] = 0

        # Show total profit
        total_profit = self.data_handler.data['Profit per day (EUR)'].sum()/self.data_handler.data_points_per_day
        print(f"Total profit: {total_profit:.2f} EUR")


    def optimize_milp(self, time_horizon: int=1, solver=cp.GUROBI, constraints_soc_end=True):
        """
        Optimize the revenue of the grid battery using a MILP solver.
        # TODO: refactor together with LP method to avoid code duplication
        Args:
            time_horizon (int): The optimization horizon in days.
            solver (cvxpy solver): The solver to use for the optimization. Must be compatible with MILP Defaults to GUROBI.
            constrain_soc_end (bool): Whether to constrain the SOC at the end of the day to be the same as at the beginning.
        """
        # Set initial SOC to 0
        previous_end_soc = 0
        
        # Number of optimization windows based on the total days and the chosen time horizon
        num_windows = (self.data_handler.total_days + time_horizon - 1) // time_horizon

        for window in range(num_windows):
            # Calculate the start and end index for the current window
            start_day = window * time_horizon
            end_day = min((window + 1) * time_horizon, self.data_handler.total_days)
            start_idx = start_day * self.data_handler.data_points_per_day
            end_idx = end_day * self.data_handler.data_points_per_day
            
            # Adjust daily_prices to span the entire time horizon
            daily_prices = self.data_handler.data['Day-ahead Price [EUR/MWh]'].values[start_idx:end_idx]
            
            # Definition of charge, discharge, is_charging, is_discharging, and soc variables
            charge = cp.Variable((end_day - start_day) * self.data_handler.data_points_per_day)
            discharge = cp.Variable((end_day - start_day) * self.data_handler.data_points_per_day)
            is_charging = cp.Variable((end_day - start_day) * self.data_handler.data_points_per_day, boolean=True)
            is_discharging = cp.Variable((end_day - start_day) * self.data_handler.data_points_per_day, boolean=True)
            soc = cp.Variable(((end_day - start_day)) * self.data_handler.data_points_per_day) #+1
            

            # Objective -> Maximize profits from buying (charging) low and selling (discharging) high
            objective = cp.Maximize(cp.sum(cp.multiply(discharge - charge, daily_prices)))
            
            start_soc = 0 if constraints_soc_end else previous_end_soc

            # Constraints
            constraints = [
                soc[0] == start_soc,  # Initial SOC is either 0 or the previous end SOC
                soc >= 0, # SOC cannot be negative
                soc <= self.battery_capacity, # SOC cannot exceed battery capacity
                charge >= 0, # Cannot charge negative amounts
                discharge >= 0, # Cannot discharge negative amounts
                charge <= self.max_power * is_charging,  # Use binary variable to enforce charging constraint
                discharge <= self.max_power * is_discharging,  # Use binary variable to enforce discharging constraint
                is_charging + is_discharging <= 1,  # Ensure battery cannot charge and discharge simultaneously
            ]

            # SOC/Battery dynamics
            for t in range((end_day - start_day) * self.data_handler.data_points_per_day):
                constraints.append(soc[t] == soc[t-1] + (charge[t] * self.efficiency) - (discharge[t] / self.efficiency))

            # Assuming the battery SOC at the end of the day should return to its initial state 
            # Assuming this to ensure continuity between days 
            # -> we can obviously change this to other assumptions or always start with the last value of the previous day
            # This is a strong assumption which might influence the outcome significatnly 
            if constraints_soc_end:
                constraints.append(soc[-1] == soc[0])

            # Solve the problem
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=solver)

            # Raise error if the problem is infeasible or unbounded
            if prob.status != "optimal":
                raise ValueError(f"Window {window} opt. propblem is {prob.status}")
            
            # After solving, directly store results in df
            self.data_handler.data.loc[start_idx:end_idx-1, 'Charge (MW)'] = charge.value
            self.data_handler.data.loc[start_idx:end_idx-1, 'Discharge (MW)'] = discharge.value
            self.data_handler.data.loc[start_idx:end_idx-1, 'SoC (MWh)'] = soc.value #[:-1]
            self.data_handler.data.loc[start_idx:end_idx-1, 'Profit per day (EUR)'] = result

            previous_end_soc = soc.value[-1]

        # Show total profit
        total_profit = self.data_handler.data['Profit per day (EUR)'].sum()/self.data_handler.data_points_per_day
        print(f"Total profit: {total_profit:.2f} EUR")


        