import os
import subprocess
import logging
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import psutil
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURABLE PARAMETERS ===

# === SUGGESTED CHANGABLE PARTS ===
CONCURRENT_PROTEINS = 12      # Number of proteins to process simultaneously, this number * CPU_CORES_PER_SMINA = TOTAL NUM of CPUs to use
CPU_CORES_PER_SMINA = 8       # Number of CPU cores each protein job will use
NUM_MODES = 2000               # The bigger this number, the more poses sampled, suggest 200 to 2000,  larger will slower the next 3 model input prepare step
SEED = 88                     # Random seed for reproducibility
SCORING_FUNCTION = 'vina'     # Scoring function to first step sampling ('vinardo', 'vina')

# === SUGGESTED FIXED PARTS ===
EXHAUSTIVENESS = 8            # Not suggested to set too high, slow and not needed
ENERGY_RANGE = 20             # Energy range for binding modes, not too small
TIMEOUT_SECONDS = 120         # Timeout in seconds (2 minutes) for each ring dock in a protein
# ===============================
TIMEOUT_LOG_FILE = "smina_timeouts.log"

def log_timeout(protein_file, ligand_file, start_time):
    """
    Log timeout information to a file.
    """
    try:
        with open(TIMEOUT_LOG_FILE, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duration = time.time() - start_time
            f.write(f"[{timestamp}] TIMEOUT - Protein: {protein_file}, Ligand: {ligand_file}, Duration: {duration:.1f}s\n")
        logger.warning(f"Logged timeout for {ligand_file} to {TIMEOUT_LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to log timeout: {e}")

def run_smina_docking_serial(args):
    """
    Run Smina sampling with configurable parameters and timeout monitoring.
    """
    protein_file, ligand_file, autobox_ligand, output_dir = args
    start_time = time.time()

    try:
        protein_name = os.path.basename(protein_file).replace('_protonated.pdb', '')
        ligand_name = os.path.basename(ligand_file).replace('.sdf', '')
        output_file = os.path.join(output_dir, f"{protein_name}_{ligand_name}_docked.sdf")

        # Use configurable number of CPU cores per Smina job
        num_cpu_per_sampling = CPU_CORES_PER_SMINA

        cmd = [
            'smina', '-r', protein_file, '-l', ligand_file,
            '--autobox_ligand', autobox_ligand, '-o', output_file,
            '--exhaustiveness', str(EXHAUSTIVENESS),
            '--seed', str(SEED),
            '--num_modes', str(NUM_MODES),
            '--energy_range', str(ENERGY_RANGE),
            '--scoring', SCORING_FUNCTION,
            '--cpu', str(num_cpu_per_sampling)
        ]

        logger.info(f"Running Smina sampling with {num_cpu_per_sampling} cores: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)

        duration = time.time() - start_time
        if result.returncode == 0:
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                logger.info(f"Smina sampling completed: {output_file} (Duration: {duration:.1f}s)")
                return output_file
            else:
                logger.error(f"Smina output file not created or empty for {ligand_file} (Duration: {duration:.1f}s)")
                log_timeout(protein_file, ligand_file, start_time)
                return None
        else:
            logger.error(f"Smina error for {ligand_file}: {result.stderr} (Duration: {duration:.1f}s)")
            log_timeout(protein_file, ligand_file, start_time)
            return None

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logger.error(f"Smina timeout for {ligand_file} after {duration:.1f}s")
        log_timeout(protein_file, ligand_file, start_time)
        return None
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Error running Smina for {ligand_file}: {e} (Duration: {duration:.1f}s)")
        log_timeout(protein_file, ligand_file, start_time)
        return None

def setup_environment():
    """Check if necessary tools (Smina) are available."""
    try:
        subprocess.run(['smina', '--help'], capture_output=True, check=True)
        logger.info("Smina is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Smina is not installed or not in PATH")
        return False

    return True

def get_system_resources():
    """Get current system resource usage."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3)
    }

def validate_config():
    """Validate configuration parameters."""
    valid_scoring_functions = ['vinardo', 'vina', 'ad4_scoring','dkoes_scoring','dkoes_fast']
    
    if SCORING_FUNCTION not in valid_scoring_functions:
        logger.error(f"Invalid scoring function: {SCORING_FUNCTION}. Valid options: {valid_scoring_functions}")
        return False
    
    if EXHAUSTIVENESS <= 0:
        logger.error(f"Exhaustiveness must be positive, got: {EXHAUSTIVENESS}")
        return False
    
    if NUM_MODES <= 0:
        logger.error(f"Num modes must be positive, got: {NUM_MODES}")
        return False
    
    if ENERGY_RANGE <= 0:
        logger.error(f"Energy range must be positive, got: {ENERGY_RANGE}")
        return False
    
    if CPU_CORES_PER_SMINA <= 0:
        logger.error(f"CPU cores per Smina job must be positive, got: {CPU_CORES_PER_SMINA}")
        return False
    
    if TIMEOUT_SECONDS <= 0:
        logger.error(f"Timeout must be positive, got: {TIMEOUT_SECONDS}")
        return False
    
    if CONCURRENT_PROTEINS <= 0:
        logger.error(f"Concurrent proteins must be positive, got: {CONCURRENT_PROTEINS}")
        return False
    
    return True

def main_sampling():
    """
    Main sampling workflow: Load tasks and run Smina sampling.
    """
    logger.info("Starting Smina sampling workflow...")
    
    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed. Please check your parameters.")
        return
    
    logger.info(f"Configuration:")
    logger.info(f"  - Concurrent proteins: {CONCURRENT_PROTEINS}")
    logger.info(f"  - CPU cores per Smina job: {CPU_CORES_PER_SMINA}")
    logger.info(f"  - Exhaustiveness: {EXHAUSTIVENESS}")
    logger.info(f"  - Num modes: {NUM_MODES}")
    logger.info(f"  - Energy range: {ENERGY_RANGE}")
    logger.info(f"  - Seed: {SEED}")
    logger.info(f"  - Scoring function: {SCORING_FUNCTION}")
    logger.info(f"  - Timeout: {TIMEOUT_SECONDS} seconds ({TIMEOUT_SECONDS/60:.1f} minutes)")
    logger.info(f"  - Timeout log file: {TIMEOUT_LOG_FILE}")

    # Load sampling tasks
    tasks_file = "docking_tasks.pkl"
    if not os.path.exists(tasks_file):
        logger.error(f"Sampling tasks file not found: {tasks_file}")
        logger.error("Please run prepare_docking_tasks.py first!")
        return

    with open(tasks_file, 'rb') as f:
        all_sampling_tasks = pickle.load(f)

    # Remove exhaustiveness from tasks since it's now a global config
    updated_tasks = []
    for task in all_sampling_tasks:
        if len(task) == 5:  # Original format with exhaustiveness
            protein_file, ligand_file, autobox_ligand, output_dir, _ = task
        else:  # New format without exhaustiveness
            protein_file, ligand_file, autobox_ligand, output_dir = task
        updated_tasks.append((protein_file, ligand_file, autobox_ligand, output_dir))
    
    all_sampling_tasks = updated_tasks

    logger.info(f"Loaded {len(all_sampling_tasks)} sampling tasks from {tasks_file}")

    total_cores = cpu_count()
    max_possible_concurrent = total_cores // CPU_CORES_PER_SMINA
    logger.info(f"System has {total_cores} CPU cores")
    logger.info(f"Maximum possible concurrent Smina jobs: {max_possible_concurrent}")
    
    resources = get_system_resources()
    logger.info(f"System resources - CPU: {resources['cpu_percent']}%, "
                f"Memory: {resources['memory_percent']}% ({resources['memory_available_gb']:.1f}GB available)")
    
    if not setup_environment():
        logger.error("Environment setup failed. Please install required tools.")
        return

    # Calculate optimal number of concurrent processes
    num_concurrent_sampling = min(len(all_sampling_tasks), max_possible_concurrent, 64)  # Cap at 64 to prevent overload

    logger.info(f"Running {len(all_sampling_tasks)} sampling tasks with {num_concurrent_sampling} concurrent processes")
    logger.info(f"Each Smina job will use {CPU_CORES_PER_SMINA} CPU cores")
    logger.info(f"Expected maximum CPU utilization: {num_concurrent_sampling * CPU_CORES_PER_SMINA} cores")
    logger.info(f"Timeout for individual jobs: {TIMEOUT_SECONDS} seconds")        
    # Initialize timeout log file
    with open(TIMEOUT_LOG_FILE, 'w') as f:
        f.write(f"Smina Timeout Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: Exhaustiveness={EXHAUSTIVENESS}, NumModes={NUM_MODES}, "
                f"EnergyRange={ENERGY_RANGE}, Seed={SEED}, Scoring={SCORING_FUNCTION}\n")
        f.write("=" * 80 + "\n")

    # Run ALL sampling tasks in parallel
    with Pool(processes=num_concurrent_sampling, maxtasksperchild=2) as pool:
        results = list(tqdm(
            pool.imap(run_smina_docking_serial, all_sampling_tasks, chunksize=1),
            total=len(all_sampling_tasks),
            desc="Running sampling jobs",
            unit="sampling"
        ))

    successful_results = [r for r in results if r is not None]
    failed_results = len(all_sampling_tasks) - len(successful_results)

    logger.info(f"Completed {len(successful_results)} out of {len(all_sampling_tasks)} sampling tasks successfully")
    logger.info(f"Failed/skipped tasks: {failed_results}")

    # Report timeout statistics
    if os.path.exists(TIMEOUT_LOG_FILE):
        timeout_count = 0
        try:
            with open(TIMEOUT_LOG_FILE, 'r') as f:
                timeout_count = len([line for line in f if 'TIMEOUT' in line])
            if timeout_count > 0:
                logger.info(f"Total timeout/skipped jobs logged: {timeout_count}")
                logger.info(f"See {TIMEOUT_LOG_FILE} for details")
        except Exception as e:
            logger.error(f"Could not read timeout log: {e}")

    final_resources = get_system_resources()
    logger.info(f"Final system resources - CPU: {final_resources['cpu_percent']}%, "
                f"Memory: {final_resources['memory_percent']}%")

    logger.info("Smina sampling workflow completed!")

if __name__ == "__main__":
    main_sampling()
