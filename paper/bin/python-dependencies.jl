using Pkg, Conda

# rebuild PyCall to ensure it links to the python provided by Conda.jl
ENV["PYTHON"] = ""
Pkg.build("PyCall")

# PyCall only works with Conda.ROOTENV
# tick requires python=3.8
Conda.add("python=3.8", Conda.ROOTENV)
Conda.add("numpy", Conda.ROOTENV)
Conda.pip_interop(true, Conda.ROOTENV)
Conda.pip("install", "tick", Conda.ROOTENV)
