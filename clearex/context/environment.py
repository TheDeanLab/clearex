import pkgutil
import IPython

def get_installed_packages():
    installed_packages = {pkg.name: pkg.module_finder.path for pkg in pkgutil.iter_modules()}
    for package_name, import_path in installed_packages.items():
        print(f"{package_name}: {import_path}")
    return installed_packages

def get_kernel_id():
    kernel_info = IPython.get_ipython().kernel
    return kernel_info.session.session
