# Debugging tensors.
def check(x, name_of_x=False):
    """Helper function for checking shapes and types during debugging."""
    print("====CHECKING======")
    if name_of_x:
        print(f"name:{name_of_x}")
    print(f"type: {type(x)}")

    if hasattr(x, "shape"):
        print(f"shape: {x.shape}")

    if hasattr(x, "len"):
        print(f"Length: {len(x)}")
    print("====END OF CHECK======")


# Seeds.
def seed_everything(seed=11711):
    "Set random seeds for python, numpy, and torch"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Create output path if missing.
def create_or_ensure_output_path(path):
    """Create the output directory if it does not exist."""
    if not os.path.exists(path):
        print(f"WARNING: Output path does not exist: {path}")
        os.makedirs(path, exist_ok=True)
        print(f"Created output directory: {path}")
        print(f"Your results will be saved to: {os.path.abspath(path)}")
