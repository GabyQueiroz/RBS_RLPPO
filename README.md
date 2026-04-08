# This code is part of the PhD Thesis in Electrical Engineering and Industrial Informatics (CPGEI - UTFPR)

Authors: **G. de Queiroz Pereira**, **D. Paulo Bertrand Renaux** and **A. E. Lazzaretti**  
Title: **COMPARATIVE ANALYSIS OF REINFORCEMENT LEARNING AND RULE-BASED SYSTEM APPROACHES FOR IRRIGATION IN HORTICULTURE**

---

# Project Structure

## **1. File: `RBS-1.py`**

This file implements the first version of the **Rule-Based System (RBS)**, responsible for calculating the irrigation schedule based on the water balance and crop parameters.

- **Functionality**:
  - Defines constants such as soil texture, crop type, irrigation efficiency, and crop coefficients (Kc) by growth stage.
  - Calculates daily irrigation requirements (`IRN` and `ITN`) using the water balance method.
  - Uses climate data (temperature, precipitation) to calculate ETo using the simplified Hargreaves equation.

- **Main Functions**:
  - `calcular_lamina_irrigacao()` — determines the daily and stage-based irrigation depth.
  - `aplicar_regras()` — applies conditional rules via a rule engine to decide whether irrigation occurs (e.g., do not irrigate if `ITN < 5`, limit if `ITN > 40`).

- **Output**:
  - Generates a table with the daily irrigation schedule containing ETo, ETc, Pe, and irrigation time.

---

## **2. File: `RBS-2.py`**

Advanced version of the **Rule-Based System (RBS)**, integrating dynamic sensor data and weather forecasts.

- **Functionality**:
  - Extends the basic RBS by integrating **forecast** and **sensor** data for real-time decisions.
  - Uses the `rule_engine` library to apply logical rules (e.g., soil and air temperature).
  - Allows hourly analysis, providing higher temporal resolution for irrigation control.

- **Data Integration**:
  - Reads files such as `dados_clima.txt` and `dados_sensor.txt`.
  - Combines actual sensor measurements (moisture, temperature, relative humidity) with daily forecasts.

- **Decision Logic**:
  - Automatically adjusts the irrigation volume.
  - Considers factors such as high temperature or dry soil to apply additional irrigation.

- **Output**:
  - Daily report with adjusted irrigation volume, environmental conditions, and qualitative explanations of decisions.

---

## **3. File: `RL-QL.py`**

Implements the **Q-Learning (QL)** algorithm integrated with the AquaCrop-OSPy model.

- **Functionality**:
  - The environment uses daily climate data (`MinTemp`, `MaxTemp`, `Precipitation`, `ReferenceET`) as state variables.
  - The Q-Learning agent learns irrigation policies that maximize productivity and water use efficiency.
  - Rewards are obtained through AquaCrop simulation, considering final crop yield.

- **Architecture**:
  - `Agent` class: defines the Q-Table, epsilon-greedy policy, and Bellman update.
  - `run_aquacrop_slice()` function: runs the AquaCrop simulation in each episode and calculates the return (dry yield).

- **Training Steps**:
  1. Loads climate data from `dados_DV.xlsx`.
  2. Runs AquaCrop simulations for each episode.
  3. Updates Q-values based on productivity and penalties for water waste.

- **Output**:
  - Trained Q-values and performance metrics per episode.

---

## **4. File: `RL-DQL.py`**

Implements the **Deep Q-Learning (DQL)** algorithm with a neural network for continuous irrigation policies.

- **Functionality**:
  - Replaces the discrete Q-Table with a neural network (`DQN`) for function approximation.
  - Learns complex, non-linear, and generalizable policies for different climate scenarios.
  - Each state contains information about precipitation, ET₀, phenological stage, and soil moisture.

- **Network Structure**:
  - Fully connected layers with ReLU activations (128–128 neurons).
  - Optimization via Adam and MSE loss function.

- **Learning Process**:
  - Uses replay memory (`deque`) to stabilize training.
  - Exploration decreases progressively with `epsilon` decay.

- **Integration**:
  - The environment interacts with AquaCrop to obtain dry yield (`Dry yield (tonne/ha)`), which generates the reward.

- **Output**:
  - Trained deep model (`DQN`) capable of estimating the optimal daily irrigation volume.

---

## **5. File: `RL-PPO.py`**

Implements the **Proximal Policy Optimization (PPO)** method — a policy gradient algorithm for continuous control integrated with AquaCrop-OSPy.

- **Functionality**:
  - The agent interacts with the realistic environment (`AquaCropEnv`), receiving states, rewards, and irrigation limits.
  - PPO learns **actor (policy)** and **critic (value)** networks to stabilize updates.
  - Includes penalties for excessive irrigation on rainy days and bonuses for water savings.

- **Model Components**:
  - **ActorNetwork**: generates the mean and standard deviation of the irrigation amount.
  - **CriticNetwork**: estimates the state value for advantage calculation.

- **Reward Function**:
  - Rewards higher productivity and penalizes excess water and water stress.
  - Includes a penalty for rain and a bonus when the agent avoids irrigating under adequate conditions.

- **Output**:
  - Trained PPO model with a smooth and efficient policy that adapts to different climate conditions.

---

## **6. File: `irrigation_soil.py`**

Integrates AquaCrop to calculate irrigation based on **soil moisture**.

- **Functionality**:
  - Monitors soil water storage and calculates moisture percentage relative to field capacity (`FC`) and permanent wilting point (`PWP`).
  - Defines custom soil layers and crop parameters.
  - Generates graphs of moisture evolution and soil water fluxes.

- **Output**:
  - Graphs and tables with moisture variation, applied depths, and water balance.

---

## **7. File: `CompareProductivity.py`**

Performs the comparison between **RBS** and **RL** strategies using AquaCrop-OSPy simulations.

- **Functionality**:
  - Loads the irrigation schedules generated by the RBS and RL methods.
  - Runs AquaCrop simulations for both strategies and calculates:
    - Fresh yield
    - Dry yield
    - Potential yield
    - Seasonal total irrigation (mm)

- **Steps**:
  1. Inserts the daily irrigation values into AquaCrop's `ITN` matrix.
  2. Runs the simulations for each method under the same climate conditions.
  3. Generates productivity versus water use graphs.

- **Output**:
  - Consolidated report comparing:
    - RL efficiency.
    - Water savings relative to RBS.
    - Productive response to the applied irrigation volume.

---

# Project Requirements

To run the experiments:

```bash
pip install aquacrop gym numpy pandas torch matplotlib seaborn rule-engine openpyxl
