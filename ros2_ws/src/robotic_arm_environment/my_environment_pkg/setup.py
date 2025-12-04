import os 
from glob import glob
from setuptools import setup

package_name = 'my_environment_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, 'my_environment_pkg.RL_algorithms_basic'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),  glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'rviz'),    glob(os.path.join('rviz', '*.rviz'))),    
        (os.path.join('share', package_name, 'worlds'),  glob(os.path.join('worlds', '*.world'))),    



    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='david',
    maintainer_email='david@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        
                           'run_environment = my_environment_pkg.run_environment:main',
                           'run_Norl_environment = my_environment_pkg.run_Norl_environment:main',
                           'data_collection = my_environment_pkg.collection_data:main',
                           'pick_and_place = my_environment_pkg.pick_and_place:main',
                           'main_rl_environment = my_environment_pkg.main_rl_environment:main',
                           'plot_metrics = my_environment_pkg.plot_metrics:main',
                           'env_markers_node = my_environment_pkg.env_markers_node:main',
                           'run_environment_D4PG = my_environment_pkg.run_environment_D4PG:main',
                          
        ],
    },  
)
