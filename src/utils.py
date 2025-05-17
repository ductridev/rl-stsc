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