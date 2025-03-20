# Standard Library Imports
import pkgutil

# Third Party Imports

# Local Imports


def get_installed_packages() -> dict:
    """Get the installed packages.

    Returns
    -------
    dict
        The installed packages.
    """
    installed_packages = {
        pkg.name: pkg.module_finder.path for pkg in pkgutil.iter_modules()
    }
    for package_name, import_path in installed_packages.items():
        print(f"{package_name}: {import_path}")
    return installed_packages
