def test_package_importable() -> None:
    """Ensures the package can be imported."""
    import dirigent
    assert dirigent.__version__ == "0.1.0"