def check(x, name_of_x=False):
    """helper function for checking shapes and types during debugging."""
    print("====CHECKING======")
    if name_of_x:
        print(f"name:{name_of_x}")
    print(f"type: {type(x)}")

    if hasattr(x, "shape"):
        print(f"shape: {x.shape}")

    if hasattr(x, "length"):
        print(f"Length: {x.length}")
    print("====END OF CHECK======")
