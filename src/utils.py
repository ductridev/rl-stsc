def set_sumo(gui, sumo_cfg_file, max_steps):
    """
    Set up the SUMO environment for the simulation.

    Args:
        gui (bool): Whether to use the GUI.
        sumo_cfg_file (str): Path to the SUMO configuration file.
        max_steps (int): Maximum number of steps for the simulation.
        sumo_port (int): Port number for SUMO.

    Returns:
        None
    """
    import os
    import sys
    import traci

    # Add SUMO tools to Python path
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

    print("Starting SUMO...")

    # Start SUMO with GUI or without GUI
    if gui:
        traci.start(["sumo-gui", "-c", os.path.join(os.getcwd(), sumo_cfg_file), "--no-step-log", "true", "--waiting-time-memory", str(max_steps), "-W", "true",  "--duration-log.disable"], label="master")
    else:
        traci.start(["sumo", "-c", os.path.join(os.getcwd(), sumo_cfg_file), "--no-step-log", "true", "--waiting-time-memory", str(max_steps), "-W", "true",  "--duration-log.disable"], label="master")

def set_train_path(model_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths

    Args:
        model_name (str): Name of the model to create a path for.
    Returns:
        str: The new model path with an incremental integer.
    """
    import os

    models_path = os.path.join(os.getcwd(), model_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path 

def set_load_model_path(model_name):
    """
    Create a model path for loading an existing model.
    Args:
        model_name (str): Name of the model to load.
    Returns:
        str: The model path for loading the existing model.
    """
    import os

    models_path = os.path.join(os.getcwd(), model_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        newest_version = str(max(previous_versions) - 1)
    data_path = os.path.join(models_path, 'model_'+newest_version, '')
    return data_path

def set_test_path(model_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths

    Args:
        model_name (str): Name of the model to create a path for.
    Returns:
        str: The new model path with an incremental integer.
    """
    import os

    models_path = os.path.join(os.getcwd(), model_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_'+new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path

def import_train_configuration(file_path):
    """
    Import the training configuration from a INI file.

    Args:
        file_path (str): Path to the INI configuration file.

    Returns:
        dict: The training configuration as a dictionary.
    """
    # Check if the file exists
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The configuration file {file_path} does not exist.")

    # Read the INI file
    import yaml

    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)

        config = {}

        # Convert the configuration to a dictionary

        # Simulation configuration
        config['gui'] = content['simulation']['gui']
        config['total_episodes'] = content['simulation']['total_episodes']
        config['max_steps'] = content['simulation']['max_steps']
        config['interphase_duration'] = content['simulation']['interphase_duration']

        # Memory configuration
        config['memory_size_min'] = content['memory']['memory_size_min']
        config['memory_size_max'] = content['memory']['memory_size_max']

        # Agent configuration
        config['agent'] = {}
        config['agent']['num_states'] = content['agent']['num_states']
        config['agent']['gamma'] = content['agent']['gamma']
        config['agent']['num_layers'] = content['agent']['model']['num_layers']
        config['agent']['batch_size'] = content['agent']['model']['batch_size']
        config['agent']['learning_rate'] = content['agent']['model']['learning_rate']
        config['agent']['decay_rate'] = content['agent']['model']['decay_rate']
        config['agent']['min_epsilon'] = content['agent']['model']['min_epsilon']
        config['agent']['epsilon'] = content['agent']['model']['epsilon']
        config['agent']['weight'] = content['agent']['weight']
        config['agent']['model'] = content['agent']['model']
        config['agent']['model']['loss_type'] = content['agent']['model']['loss_type']
        # # Green duration agent
        # config['green_duration_agent'] = {}
        # config['green_duration_agent']['num_states'] = content['green_duration_agent']['num_states']
        # config['green_duration_agent']['num_actions'] = content['green_duration_agent']['num_actions']
        # config['green_duration_agent']['gamma'] = content['green_duration_agent']['gamma']
        # config['green_duration_agent']['num_layers'] = content['green_duration_agent']['model']['num_layers']
        # config['green_duration_agent']['batch_size'] = content['green_duration_agent']['model']['batch_size']
        # config['green_duration_agent']['learning_rate'] = content['green_duration_agent']['model']['learning_rate']
        # config['green_duration_agent']['actions_space'] = content['green_duration_agent']['model']['actions_space']
        
        # # Selector phase agent
        # config['selector_phase_agent'] = {}
        # config['selector_phase_agent']['num_states'] = content['selector_phase_agent']['num_states']
        # config['selector_phase_agent']['num_actions'] = content['selector_phase_agent']['num_actions']
        # config['selector_phase_agent']['gamma'] = content['selector_phase_agent']['gamma']
        # config['selector_phase_agent']['num_layers'] = content['selector_phase_agent']['model']['num_layers']
        # config['selector_phase_agent']['batch_size'] = content['selector_phase_agent']['model']['batch_size']
        # config['selector_phase_agent']['learning_rate'] = content['selector_phase_agent']['model']['learning_rate']

        # Traffic demand configuration
        config['vehicle_counts'] = content['vehicle_counts']
        config['vehicle_counts']['low'] = content['vehicle_counts']['low']
        config['vehicle_counts']['medium'] = content['vehicle_counts']['medium']
        config['vehicle_counts']['high'] = content['vehicle_counts']['high']

        # Random demand
        config["edge_groups"] = content["edge_groups"]

        # Training configuration
        config['training_epochs'] = content['model']['training_epochs']
        config['save_model_name'] = content['model'].get('save_model_name', 'dqn_model')
        config['load_model_name'] = content['model'].get('load_model_name', None)
        config['load_q_table_name'] = content['model'].get('load_q_table_name', None)
        config['save_interval'] = content['model'].get('save_interval', 10)
        config['models_path_name'] = content['dir']['models_path_name']
        config['sumo_cfg_file'] = content['dir']['sumocfg_file_name']
        config['training_steps'] = content['model']['training_steps']
        config['updating_target_network_steps'] = content['model']['updating_target_network_steps']

        # Intersection configuration
        config['traffic_lights'] = content['traffic_lights']

        # Accident configuration
        config["junction_id_list"] = [junction["id"] for junction in content['accident']['junction']]
        config["start_step"] = content['accident']['start_step']
        config["duration"] = content['accident']['duration']

        print("Configuration loaded successfully.")
        return config
    
def import_test_configuration(file_path):
    """
    Import the testing configuration from a INI file.

    Args:
        file_path (str): Path to the INI configuration file.

    Returns:
        dict: The testing configuration as a dictionary.
    """
    # Check if the file exists
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The configuration file {file_path} does not exist.")

    # Read the INI file
    import yaml

    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)

        config = {}

        # Convert the configuration to a dictionary
        # Simulation configuration
        config['gui'] = content['simulation']['gui']
        config['max_steps'] = content['simulation']['max_steps']
        config['episode_seed'] = content['simulation']['episode_seed']
        config['interphase_duration'] = content['simulation']['interphase_duration']

        config['agent']['num_states'] = content['agent']['num_states']
        # # Green duration agent configuration
        # config['green_duration_agent']['num_states'] = content['green_duration_agent']['num_states']
        # config['green_duration_agent']['num_actions'] = content['green_duration_agent']['num_actions']

        # # Selector phase agent
        # config['selector_phase_agent']['num_states'] = content['selector_phase_agent']['num_states']
        # config['selector_phase_agent']['num_actions'] = content['selector_phase_agent']['num_actions']

        # Testing configuration
        config['models_path_name'] = content['dir']['models_path_name']
        config['sumo_cfg_file'] = content['dir']['sumocfg_file_name']
        config['model_to_test'] = content['dir']['model_to_test']

        # Intersection configuration
        config['traffic_lights'] = content['traffic_lights']

        return config