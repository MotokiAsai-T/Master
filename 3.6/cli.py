import argparse
import json
from hello import main_function  # Assuming main_function is the entry point for the main logic

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the demand matching algorithm.")
    parser.add_argument('--Num_time', type=int, default=5, help='Number of time periods (default: 5)')
    parser.add_argument('--Total_vehicles', type=int, default=10, help='Total number of vehicles (default: 10)')
    parser.add_argument('--p_inc', type=float, default=0.6, help='Probability increment (default: 0.6)')
    parser.add_argument('--base_cost', type=float, default=10.0, help='Base cost (default: 10.0)')
    parser.add_argument('--base_penalty', type=float, default=50.0, help='Base penalty (default: 50.0)')
    parser.add_argument('--variation', type=float, default=0.2, help='Variation for cost/penalty (default: 0.2)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility (optional)')
    parser.add_argument('--transition', type=str, help='Transition matrix in JSON format (optional)')
    parser.add_argument('--theta1', type=float, default=1.0, help='Algorithm specific parameter (default: 1.0)')
    parser.add_argument('--theta2', type=float, default=1.0, help='Algorithm specific parameter (default: 1.0)')
    parser.add_argument('--start_state', type=str, default='state_0_0', help='Initial state label (default: "state_0_0")')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Convert transition matrix from JSON string if provided
    if args.transition:
        args.transition = json.loads(args.transition)

    # Call the main function from hello.py with parsed arguments
    main_function(args)

if __name__ == "__main__":
    main()