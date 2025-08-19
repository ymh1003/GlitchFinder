from setuptools import setup

setup(
    name='Gcode-Checking-Project',
    version='0.0.1',
    py_modules=['gcode_comp_Z', 'timerutils'],
    install_requires=[],
    scripts=['gcode_comp_Z.py', 'heatmap_merge.py']
)
