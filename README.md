# NUMN21_optimization_project

## Contributions

Linn Preuss Jelvez, Lukas Nord, Scott Gibson, Orsolya Bosáková.

Each team member has done the code individually and then we combined them into one coherent document.

## Usage

Simply install requirements, and run the `main.py` file. Optionally, to generate the fractal found in the `imgs` folder, run the `parallell_fractals.py` file.

## Plotting

Contour plots for the Rosenbrock functions.

Grid plots for the Hessian approximations from Quasi-Newton methods. The current comparison is the approximation from finite differences, but another alternative could be from `scipy.differentiate.hessian`, but that wasn't immediately compatible with the Chebyquad functions.

## Images/Outputs

1. Rosenbrock path and Hessian error for:
    - Classical Newton
    - Inexact line search
    - Bad Broyden
    - Good Broyden
    - Symmetric Broyden
    - DFP
    - BFGS
The output contains the number of steps taken. 

#### Classical Newton 
<img width="49%" height="584" alt="Screenshot 2026-10-12 at 22 50 58" src="https://github.com/user-attachments/assets/a68d0f07-50a0-442b-8eb9-2b2d55c87b82" />
<img width="49%" height="482" alt="Screenshot 2026-10-12 at 22 50 40" src="https://github.com/user-attachments/assets/8844e49c-da3a-48a7-a013-8c03068c6a8b" />


#### Inexact line search
<img width="49%" height="586" alt="Screenshot 2026-10-12 at 22 30 45" src="https://github.com/user-attachments/assets/56eea838-674e-4ca1-a3ff-762628db78c1" />
<img width="49%" height="486" alt="Screenshot 2026-10-12 at 22 30 20" src="https://github.com/user-attachments/assets/a0e97745-29fc-46e6-a8d0-eba312925310" />

#### Bad Broyden
<img width="49%" height="584" alt="Screenshot 2026-10-12 at 22 32 13" src="https://github.com/user-attachments/assets/c6c923b4-ce2e-4ff0-ab98-6ebcec81fb8f" />
<img width="49%" height="492" alt="Screenshot 2026-10-12 at 22 31 54" src="https://github.com/user-attachments/assets/435451ae-f612-4eea-84ea-694d30b68db8" />

#### Good Broyden
<img width="49%" height="586" alt="Screenshot 2026-10-15 at 02 38 51" src="https://github.com/user-attachments/assets/717897a3-ca50-41ae-85ed-4f7e35c1b301" />
<img width="49%" height="480" alt="Screenshot 2026-10-15 at 02 38 31" src="https://github.com/user-attachments/assets/2092758a-b680-4046-900a-5233efe587a4" />


#### Symmetric Broyden
<img width="49%" height="582" alt="Screenshot 2026-10-15 at 02 39 41" src="https://github.com/user-attachments/assets/8894cdcb-a8c2-4da7-a221-efc79fee7bf3" />
<img width="49%" height="481" alt="Screenshot 2026-10-15 at 02 39 25" src="https://github.com/user-attachments/assets/ef15d8b4-37bb-40ff-981a-9eeadf92c0af" />


#### DFP
<img width="49%" height="580" alt="Screenshot 2026-10-15 at 02 40 31" src="https://github.com/user-attachments/assets/ef5dc791-4301-4215-a207-12a2a26f26fc" />
<img width="49%" height="477" alt="Screenshot 2026-10-15 at 02 40 18" src="https://github.com/user-attachments/assets/1a01b62c-fd1a-432b-aa0d-6c8f225211e2" />


#### BFGS
<img width="49%" height="582" alt="Screenshot 2026-10-15 at 02 41 31" src="https://github.com/user-attachments/assets/226e4004-0ecb-4998-a66f-edb32ba60c7c" />
<img width="49%" height="481" alt="Screenshot 2026-10-15 at 02 41 15" src="https://github.com/user-attachments/assets/bab24c77-08b1-4536-b41b-04270761498e" />


## TODO/IMPROVEMENTS

