import subprocess

# Replace 'other_program.py' with the name of the Python program you want to run
rf = 'Random_Forest_2.py'
xg = 'XGBoost New.py'
mlp = 'MLP 2.py'
lgb = 'LightGBM3.py'
app = 'Always pick plaintiff.py'

subroutines = [rf, xg, lgb, mlp, app]

for subroutine in subroutines:
    # Run the other Python program and capture the output
    try:
        result = subprocess.run(['python', subroutine], check=True, capture_output=True, text=True)
        #print("Output from the subprocess:")
        #print(result.stdout)
        with open('output.txt', 'a') as file:
            file.write(str(result.stdout))
            file.write('\n\n\n')
            print("Subroutine: ", subroutine)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("Output from the subprocess (if available):")
        print(e.stdout)  # Print the output even if there's an error
#print("Returning Control")
