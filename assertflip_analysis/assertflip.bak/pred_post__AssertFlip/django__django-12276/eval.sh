#!/bin/bash
set -uxo pipefail
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git diff HEAD 53d8646f799de7f92ab9defe9dc56c6125448102 >> /root/pre_state.patch
sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
export LANG=en_US.UTF-8
export LANGUAGE=en_US:en
export LC_ALL=en_US.UTF-8
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git diff 53d8646f799de7f92ab9defe9dc56c6125448102
source /opt/miniconda3/bin/activate
conda activate testbed
python -m pip install -e .
git apply -v - <<'EOF_114329324912'
diff --git a/django/forms/widgets.py b/django/forms/widgets.py
--- a/django/forms/widgets.py
+++ b/django/forms/widgets.py
@@ -387,6 +387,9 @@ def value_from_datadict(self, data, files, name):
     def value_omitted_from_data(self, data, files, name):
         return name not in files
 
+    def use_required_attribute(self, initial):
+        return super().use_required_attribute(initial) and not initial
+
 
 FILE_INPUT_CONTRADICTION = object()
 
@@ -451,9 +454,6 @@ def value_from_datadict(self, data, files, name):
             return False
         return upload
 
-    def use_required_attribute(self, initial):
-        return super().use_required_attribute(initial) and not initial
-
     def value_omitted_from_data(self, data, files, name):
         return (
             super().value_omitted_from_data(data, files, name) and

EOF_114329324912
git apply -v - <<'EOF_114329324912'
diff --git a/dev/null b/tests/test_coverup_django__django-12276.py
new file mode 100644
index e69de29..98b227c 100644
--- /dev/null
+++ b/tests/test_coverup_django__django-12276.py
@@ -0,0 +1,18 @@
+from django import forms
+from django.test import SimpleTestCase
+
+class TestFileInputRequiredAttribute(SimpleTestCase):
+    def test_file_input_with_initial_data_shows_required(self):
+        class TestForm(forms.Form):
+            file_field = forms.FileField(widget=forms.ClearableFileInput(), required=True)
+
+        # Mock initial data with a file object that has a 'url' attribute
+        initial_data = {'file_field': type('MockFile', (object,), {'url': 'mock_url'})()}
+
+        # Create form with initial data
+        form_with_initial = TestForm(initial=initial_data)
+        rendered_html_with_initial = form_with_initial.as_p()
+
+        # Check if 'required' attribute is present in the HTML output
+        # The test should fail if 'required' is present, exposing the bug
+        self.assertIn('required', rendered_html_with_initial)  # This should fail, revealing the bug

EOF_114329324912
python3 /root/trace.py --timing --trace --count -C coverage.cover --include-pattern '/testbed/(django/forms/widgets\.py)' ./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 test_coverup_django__django-12276
cat coverage.cover
git checkout 53d8646f799de7f92ab9defe9dc56c6125448102
git apply /root/pre_state.patch
