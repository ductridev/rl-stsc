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

    # Set environment variables
    os.environ["SUMO_HOME"] = "/usr/local/share/sumo"
    os.environ["SUMO_GUI"] = "1" if gui else "0"

    # Add SUMO tools to Python path
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

    # Start SUMO with GUI or without GUI
    if gui:
        traci.start(["sumo-gui", "-c", os.path.join('intersection', sumo_cfg_file), "--no-step-log", "true", "--waiting-time-memory", max_steps])
    else:
        traci.start(["sumo", "-c", os.path.join('intersection', sumo_cfg_file), "--no-step-log", "true", "--waiting-time-memory", max_steps])

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
        config['gui'] = content['simulation']['gui']
        config['total_episodes'] = content['simulation']['total_episodes']
        config['max_steps'] = content['simulation']['max_steps']
        config['n_cars_generated'] = content['simulation']['n_cars_generated']
        config['green_duration'] = content['simulation']['green_duration']
        config['yellow_duration'] = content['simulation']['yellow_duration']
        config['num_layers'] = content['model']['num_layers']
        config['width_layers'] = content['model']['width_layers']
        config['batch_size'] = content['model']['batch_size']
        config['learning_rate'] = content['model']['learning_rate']
        config['training_epochs'] = content['model']['training_epochs']
        config['memory_size_min'] = content['memory']['memory_size_min']
        config['memory_size_max'] = content['memory']['memory_size_max']
        config['num_states'] = content['agent']['num_states']
        config['num_actions'] = content['agent']['num_actions']
        config['gamma'] = content['agent']['gamma']
        config['models_path_name'] = content['dir']['models_path_name']
        config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
        config['intersections'] = content['intersections']

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
        config['gui'] = content['simulation']['gui']
        config['total_episodes'] = content['simulation']['total_episodes']
        config['max_steps'] = content['simulation']['max_steps']
        config['n_cars_generated'] = content['simulation']['n_cars_generated']
        config['green_duration'] = content['simulation']['green_duration']
        config['yellow_duration'] = content['simulation']['yellow_duration']
        config['num_layers'] = content['model']['num_layers']
        config['width_layers'] = content['model']['width_layers']
        config['batch_size'] = content['model']['batch_size']
        config['learning_rate'] = content['model']['learning_rate']
        config['training_epochs'] = content['model']['training_epochs']
        config['memory_size_min'] = content['memory']['memory_size_min']
        config['memory_size_max'] = content['memory']['memory_size_max']
        config['num_states'] = content['agent']['num_states']
        config['num_actions'] = content['agent']['num_actions']
        config['gamma'] = content['agent']['gamma']
        config['models_path_name'] = content['dir']['models_path_name']
        config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
        config['intersections'] = content['intersections']

        return config