from setuptools import setup

package_name = 'agri_swarm_core'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Vishal Selvarasu',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'detector_node = agri_swarm_core.detector_node:main',
            'generate_field = agri_swarm_core.generate_field:main',
            'lane_follower_node = agri_swarm_core.lane_follower_node:main',
        ],
    },
)
