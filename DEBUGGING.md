# Debugging Guide

This guide explains how to debug LCServer processing steps using various techniques.

## Table of Contents

1. [Command-Line Debugging](#command-line-debugging)
2. [IDE Debugging](#ide-debugging)
3. [Interactive Python Debugging](#interactive-python-debugging)
4. [Common Debugging Scenarios](#common-debugging-scenarios)

---

## Command-Line Debugging

### Option 1: Post-Mortem Debugging with --pdb

The `--pdb` flag automatically drops into the Python debugger when an exception occurs.

```bash
# Drop into pdb on error
python manage.py test_target 1 -s info --pdb --verbose
```

**What happens:**
1. Processing runs normally
2. When an exception occurs, you see the traceback
3. You're automatically dropped into pdb at the point of failure
4. You can inspect variables, step through code, etc.

**Common pdb commands:**
```
l          # List source code around current line
ll         # List entire function
p var      # Print variable
pp var     # Pretty-print variable
w          # Show stack trace (where you are)
u          # Move up the stack
d          # Move down the stack
args       # Show function arguments
!code      # Execute arbitrary Python code
c          # Continue execution
q          # Quit debugger
```

**Example session:**
```bash
$ python manage.py test_target 1 -s info --pdb --verbose

Running step: info
============================================================

Acquiring the info on RY Cnc

✗ Step 'info' failed: 'MAIN_ID'
Traceback (most recent call last):
  File ".../processing/info.py", line 163, in target_info
    log(f"{r['MAIN_ID']} = {r['OTYPE']}")
KeyError: 'MAIN_ID'

Entering post-mortem debugger...
> /path/to/processing/info.py(163)target_info()
-> log(f"{r['MAIN_ID']} = {r['OTYPE']}")

(Pdb) l
158         # Query SIMBAD
159         r = Simbad.query_object(target.name)
160
161         if r is not None:
162             for row in r:
163 ->          log(f"{row['MAIN_ID']} = {row['OTYPE']}")
164
(Pdb) p r
<Table length=1>
  TYPED_ID    ... FLUX_V  FLUX_ERROR_V
   object     ...        ...
----------- ... ------- -------------
    RY Cnc  ...    --            --

(Pdb) p r.colnames
['TYPED_ID', 'RA', 'DEC', 'RA_PREC', 'DEC_PREC', 'COO_ERR_MAJA', ...]

(Pdb) # Ah! The column is called 'TYPED_ID' not 'MAIN_ID'
(Pdb) p row['TYPED_ID']
'RY Cnc'

(Pdb) q
```

### Option 2: Debug Mode with --debug

The `--debug` flag disables all exception handling, allowing exceptions to propagate naturally. This is useful with IDE debuggers or when you want to see the raw exception.

```bash
# Let exceptions propagate (for IDE debugging)
python manage.py test_target 1 -s info --debug --verbose
```

**When to use:**
- Running from an IDE with breakpoints set
- Want to see the exact line where the error occurs
- Need full control over exception handling

**Example:**
```bash
$ python manage.py test_target 1 -s info --debug --verbose

Running step: info
============================================================

Starting step: info
Target: RY Cnc
Path: /Users/.../targets/1

Acquiring the info on RY Cnc

Traceback (most recent call last):
  File "manage.py", line 22, in <module>
    main()
  [... full stack trace ...]
  File "/path/to/processing/info.py", line 163, in target_info
    log(f"{r['MAIN_ID']} = {r['OTYPE']}")
KeyError: 'MAIN_ID'
```

### Option 3: Manual Breakpoints

Insert `breakpoint()` or `pdb.set_trace()` directly in the processing code.

**Edit the source's module in `lcserver/processing/`:**
```python
def target_info(config, basepath='.', verbose=None, show=False):
    # ... existing code ...

    # Add breakpoint before the problematic line
    import pdb; pdb.set_trace()

    if r is not None:
        for row in r:
            log(f"{row['MAIN_ID']} = {row['OTYPE']}")
```

**Then run normally:**
```bash
python manage.py test_target 1 -s info --verbose
```

---

## IDE Debugging

### PyCharm

1. **Set up Run Configuration:**
   - Run → Edit Configurations
   - Add new "Django Server" configuration
   - Script path: `/path/to/manage.py`
   - Parameters: `test_target 1 -s info --debug --verbose`
   - Working directory: `/path/to/lcserver`

2. **Set Breakpoints:**
   - Open the source's module in `lcserver/processing/`
   - Click in the gutter next to the line you want to debug
   - Red dot appears

3. **Start Debug Session:**
   - Click the Debug icon (bug icon)
   - Execution stops at your breakpoint
   - Use debugger panel to:
     - Step over (F8)
     - Step into (F7)
     - Inspect variables
     - Evaluate expressions

### VSCode

1. **Create launch.json:**

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug test_target",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/manage.py",
            "args": [
                "test_target",
                "1",
                "-s", "info",
                "--debug",
                "--verbose"
            ],
            "django": true,
            "console": "integratedTerminal"
        }
    ]
}
```

2. **Set Breakpoints:**
   - Click in the gutter next to any line
   - Red dot appears

3. **Start Debugging:**
   - Press F5 or click "Run and Debug"
   - Execution stops at breakpoints

---

## Interactive Python Debugging

### Django Shell with IPython

Start an interactive shell with access to all Django models and functions:

```bash
# Install ipython for better experience
pip install ipython

# Start Django shell
python manage.py shell
```

**Example debugging session:**
```python
# Import necessary modules
from lcserver import models, processing
from functools import partial
import os

# Get target
target = models.Target.objects.get(id=1)
print(f"Target: {target.name}")

# Setup config
config = target.config
config['target_name'] = target.name
basepath = target.path()

# Create logging function
def log(*args, **kwargs):
    print(' '.join(str(arg) for arg in args))

# Now you can step through the processing manually
# For example, let's debug target_info:

# Import the resolve function
from stdpipe import resolve

# Try resolving the target
target_obj = resolve.resolve(target.name, maxdist=1./60)
print(f"Resolved: {target_obj}")
print(f"RA: {target_obj.ra if target_obj else 'None'}")
print(f"Dec: {target_obj.dec if target_obj else 'None'}")

# Try SIMBAD query
from astroquery.simbad import Simbad
r = Simbad.query_object(target.name)
print(f"SIMBAD result columns: {r.colnames if r else 'None'}")

# Inspect the result
if r is not None:
    for row in r:
        print(f"Available columns: {row.keys()}")
        # Now you can check which column has the data you need
        if 'TYPED_ID' in row:
            print(f"TYPED_ID: {row['TYPED_ID']}")
        if 'MAIN_ID' in row:
            print(f"MAIN_ID: {row['MAIN_ID']}")

# Call the actual processing function with debugging
processing.target_info(config, basepath=basepath, verbose=log, show=False)
```

### Using IPython's %debug Magic

```bash
python manage.py shell
```

```python
from lcserver import models, processing

target = models.Target.objects.get(id=1)
config = target.config
config['target_name'] = target.name

# Try to run the function
try:
    processing.target_info(config, basepath=target.path())
except Exception:
    # Automatically drop into debugger at exception point
    import IPython
    IPython.embed()
```

---

## Common Debugging Scenarios

### Scenario 1: API Returns Unexpected Data

**Problem:** External API (SIMBAD, ZTF, etc.) returns data in different format than expected.

**Solution 1 - Interactive inspection:**
```bash
python manage.py shell
```

```python
from astroquery.simbad import Simbad

# Query the API directly
result = Simbad.query_object("RY Cnc")

# Inspect the result
print(f"Type: {type(result)}")
print(f"Columns: {result.colnames if result else 'None'}")
print(f"Length: {len(result) if result else 0}")

# Print first row
if result:
    row = result[0]
    print(f"Row keys: {row.keys()}")
    for key in row.keys():
        print(f"  {key}: {row[key]}")
```

**Solution 2 - Add debug output:**
```bash
# Edit the source's module in `lcserver/processing/` to add debug prints before the error
python manage.py test_target 1 -s info --verbose
```

### Scenario 2: File Not Created

**Problem:** Expected output file is not generated.

**Solution:**
```bash
# Run with verbose mode to see all output
python manage.py test_target 1 -s ztf --verbose

# Check the log file
cat targets/1/ztf.log

# Use --pdb to inspect at the point of failure
python manage.py test_target 1 -s ztf --pdb --verbose
```

### Scenario 3: Coordinate Resolution Fails

**Problem:** Target name doesn't resolve to coordinates.

**Solution:**
```bash
python manage.py shell
```

```python
from stdpipe import resolve

# Test different names
names = ["RY Cnc", "RY Cancer", "RY Cancri", "RA=08:34:19.0 Dec=+10:51:52"]

for name in names:
    target = resolve.resolve(name, maxdist=1./60)
    print(f"{name}: {target}")
    if target:
        print(f"  RA: {target.ra.deg}, Dec: {target.dec.deg}")
```

### Scenario 4: Memory or Performance Issues

**Problem:** Processing is slow or runs out of memory.

**Solution 1 - Profile memory usage:**
```python
# Add to the source's module
import tracemalloc

def target_ztf(config, basepath='.', verbose=None, show=False):
    tracemalloc.start()

    # ... existing code ...

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory: {current / 10**6:.1f} MB")
    print(f"Peak memory: {peak / 10**6:.1f} MB")
    tracemalloc.stop()
```

**Solution 2 - Profile execution time:**
```python
import time

def target_ztf(config, basepath='.', verbose=None, show=False):
    start = time.time()

    # ... existing code ...

    elapsed = time.time() - start
    print(f"Execution time: {elapsed:.2f} seconds")
```

### Scenario 5: Debugging Celery Tasks

**Problem:** Need to debug code running through Celery.

**Solution 1 - Use test_target instead:**
```bash
# Don't use Celery - run synchronously
python manage.py test_target 1 -s info --debug --verbose
```

**Solution 2 - Debug Celery task directly:**
```bash
# Start Celery with single worker in foreground
celery -A lcserver worker --loglevel=debug --concurrency=1

# In another terminal, trigger the task
python manage.py shell
```

```python
from lcserver.celery_tasks import task_info
result = task_info(1)
```

---

## Tips and Best Practices

1. **Start Simple:**
   - Use `--verbose` first to see what's happening
   - Add `--pdb` if you need to inspect state
   - Use `--debug` for IDE debugging

2. **Use Django Shell for Exploration:**
   - Test API calls interactively
   - Inspect data structures
   - Try different approaches

3. **Add Temporary Debug Output:**
   - Add `print()` statements in the source's module
   - Use `verbose()` function for logging
   - Remember to remove debug code after fixing

4. **Check Logs:**
   - All steps log to `targets/{id}/{step}.log`
   - Logs show exact error messages and tracebacks
   - Compare successful vs failed runs

5. **Isolate the Problem:**
   - Run single steps with `-s step_name`
   - Test with known-good targets first
   - Simplify until you find the minimal failing case

6. **Document Findings:**
   - Note which API calls fail
   - Record unexpected data formats
   - Share solutions with team

---

## Quick Reference

```bash
# Post-mortem debugging
python manage.py test_target ID -s STEP --pdb --verbose

# IDE debugging (PyCharm, VSCode)
python manage.py test_target ID -s STEP --debug --verbose

# Manual inspection
python manage.py shell

# View logs
cat targets/ID/STEP.log

# List available targets
python manage.py targets --list

# Show target details
python manage.py targets --info ID

# Check system status
python manage.py maintenance --check
```

---

## Getting Help

If you're still stuck:

1. Check the logs in `targets/{id}/*.log`
2. Run with `--verbose` to see full output
3. Use `--pdb` to inspect the exact state at failure
4. Review the *Management commands* section of [README.md](README.md)
5. Check the module under `lcserver/processing/` for the specific step
6. Test external APIs directly in Django shell
