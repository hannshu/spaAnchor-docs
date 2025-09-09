Installation
------------

We highly recommend using `Anaconda <https://www.anaconda.com/>`__ 
to manage your Python environment and packages, as it 
significantly simplifies dependency resolution. (OPTIONAL)

.. code-block:: bash

   conda create -n spaanchor_env
   conda activate spaanchor_env

Download and install the spaAnchor source code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The source code of spaAnchor can be find at our `GitHub
repository <https://github.com/YuanSTLab/spaanchor>`__.

.. code:: bash

   git clone https://github.com/YuanSTLab/spaAnchor
   cd spaAnchor

spaAnchor can be executed on both CPU and GPU hardware. 
For optimal performance, **we strongly recommend running spaAnchor on a GPU**.

Please follow the appropriate installation steps below 
based on your desired hardware:

**1. For GPU (Recommended):**

.. code-block:: bash

   pip install .[gpu]

**2. For CPU:**

.. code-block:: bash

    pip install .[cpu]

.. note::

   ``spaAnchor``'s main installation does not pin a specific version for the dependencies. 
   This is designed to maximize compatibility and minimize installation conflicts. 
   By specifying flexible version requirements, we reduce the risk of breaking 
   your existing environment, as you may already have packages like ``numpy`` or 
   ``scanpy`` installed for other projects.

   For most use cases, using slightly different versions of these dependencies 
   is not expected to impact the functionality or performance of ``spaAnchor``.

   However, for full reproducibility, we provide a ``requirements.txt`` file. 
   This file contains the exact version of the core dependencies.

   To use this file, we strongly recommend creating a new Anaconda environment 
   to avoid any conflicts with your existing packages.

   .. code-block:: bash

      conda create -n spaanchor-repro python=3.9.21
      conda activate spaanchor-repro
      pip install -r requirements.txt
      pip install .


For diagonal integration
~~~~~~~~~~~~~~~~~~~~~~~~

Once spaAnchor is set up, user can use it by importing it directly. 
For diagonal integration, please follow the instructions below to 
download the HIPT pre-trained parameters, or download it manually.

.. code:: bash

   wget https://github.com/mahmoodlab/HIPT/raw/refs/heads/master/HIPT_4K/Checkpoints/vit256_small_dino.pth 
