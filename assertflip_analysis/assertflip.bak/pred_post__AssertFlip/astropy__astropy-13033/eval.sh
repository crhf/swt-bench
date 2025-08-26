#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git diff HEAD 298ccb478e6bf092953bca67a3d29dc6c35f6752 >> /root/pre_state.patch
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff 298ccb478e6bf092953bca67a3d29dc6c35f6752
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .[test] --verbose
git apply -v - <<'EOF_114329324912'
diff --git a/astropy/timeseries/core.py b/astropy/timeseries/core.py
--- a/astropy/timeseries/core.py
+++ b/astropy/timeseries/core.py
@@ -55,6 +55,13 @@ class BaseTimeSeries(QTable):
     _required_columns_relax = False
 
     def _check_required_columns(self):
+        def as_scalar_or_list_str(obj):
+            if not hasattr(obj, "__len__"):
+                return f"'{obj}'"
+            elif len(obj) == 1:
+                return f"'{obj[0]}'"
+            else:
+                return str(obj)
 
         if not self._required_columns_enabled:
             return
@@ -76,9 +83,10 @@ def _check_required_columns(self):
 
             elif self.colnames[:len(required_columns)] != required_columns:
 
-                raise ValueError("{} object is invalid - expected '{}' "
-                                 "as the first column{} but found '{}'"
-                                 .format(self.__class__.__name__, required_columns[0], plural, self.colnames[0]))
+                raise ValueError("{} object is invalid - expected {} "
+                                 "as the first column{} but found {}"
+                                 .format(self.__class__.__name__, as_scalar_or_list_str(required_columns),
+                                            plural, as_scalar_or_list_str(self.colnames[:len(required_columns)])))
 
             if (self._required_columns_relax
                     and self._required_columns == self.colnames[:len(self._required_columns)]):

EOF_114329324912
git apply -v - <<'EOF_114329324912'
diff --git a/dev/null b/astropy/tests/test_coverup_astropy__astropy-13033.py
new file mode 100644
index e69de29..933cf7c 100644
--- /dev/null
+++ b/astropy/tests/test_coverup_astropy__astropy-13033.py
@@ -0,0 +1,19 @@
+import pytest
+from astropy.time import Time
+from astropy.timeseries import TimeSeries
+import numpy as np
+
+def test_misleading_exception_message():
+    # Create a TimeSeries object with required columns 'time' and 'flux'
+    time = Time(np.arange(100000, 100003), format='jd')
+    ts = TimeSeries(time=time, data={"flux": [99.9, 99.8, 99.7]})
+    ts._required_columns = ["time", "flux"]
+
+    # Temporarily disable required column checks to remove 'flux'
+    ts._required_columns_enabled = False
+    ts.remove_column("flux")
+    ts._required_columns_enabled = True
+
+    # Access the TimeSeries object to trigger the column check
+    with pytest.raises(ValueError, match="required columns are missing"):
+        ts._check_required_columns()

EOF_114329324912
python3 /root/trace.py --timing --trace --count -C coverage.cover --include-pattern '/testbed/(astropy/timeseries/core\.py)' -m pytest --no-header -rA  -p no:cacheprovider astropy/tests/test_coverup_astropy__astropy-13033.py
cat coverage.cover
git checkout 298ccb478e6bf092953bca67a3d29dc6c35f6752
git apply /root/pre_state.patch
