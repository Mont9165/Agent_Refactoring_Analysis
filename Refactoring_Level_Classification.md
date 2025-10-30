Refactoring Classification

Definitions of Refactoring Levels

This classification model is based on the research by Murphy-Hill et al. and categorizes refactorings based on their impact on signatures and code blocks.

* High-level (H)  Refactorings that only change the signatures of classes, methods, or fields. These changes alter the "public interface" and often require modifications to the code that calls or references the changed element.
* Medium-level (M) ↔ Refactorings that change both the signature and significantly change a code block. These operations bridge the gap between internal logic and external structure.
* Low-level (L)  Refactorings where changes are confined exclusively to a code block (e.g., inside a method body) and are not visible from the outside.


High-level (H) Refactorings (58 types)

These refactorings only change the signatures of classes, methods, or fields.

* Add Class Annotation: Adds a metadata annotation to a class declaration.
* Add Class Modifier: Adds a modifier like static, final, or abstract from a class.
* Add Attribute Annotation: Adds an annotation to a class field (attribute).
* Add Attribute Modifier: Adds a modifier like static, final, or volatile from a field.
* Add Method Annotation: Adds an annotation to a method declaration.
* Add Method Modifier: Adds a modifier like static, final, or abstract from a method.
* Add Parameter: Adds a new parameter to a method's signature.
* Add Parameter Annotation: Adds an annotation to a method parameter.
* Add Thrown Exception Type: Adds a new exception to a method's throws clause.
* Change Class Access Modifier: Changes the visibility of a class (e.g., public to package-private).
* Change Attribute Access Modifier: Changes the visibility of a field (e.g., private to public).
* Change Method Access Modifier: Changes the visibility of a method.
* Change Thrown Exception Type: Modifies an existing exception in a method's throws clause.
* Change Type Declaration Kind: Changes a type's declaration (e.g., from a class to an interface).
* Collapse Hierarchy: Merges a class and its superclass.
* Encapsulate Attribute: Makes a public field private and provides public getter and setter methods.
* Extract Interface: Creates a new interface from a subset of methods in an existing class.
* Extract Superclass: Creates a new superclass from features in one or more existing classes.
* Merge Attribute: Combines multiple fields into a single field.
* Merge Class: Combines the features of two or more classes into a single class.
* Merge Package: Merges multiple packages into a single one.
* Merge Parameter: Combines multiple parameters into a single parameter object.
* Modify Class Annotation: Changes the values of an existing annotation on a class.
* Modify Attribute Annotation: Changes an existing annotation on a field.
* Modify Method Annotation: Changes an existing annotation on a method.
* Modify Parameter Annotation: Changes an existing annotation on a parameter.
* Move Attribute: Relocates a field to a different class.
* Move Class: Moves a class to a different package.
* Move Method: Relocates a method to a different class.
* Move Package: Relocates a package within the project structure.
* Move and Rename Attribute: A combination of moving a field to another class and renaming it.
* Move and Rename Class: A combination of moving a class to a new package and renaming it.
* Move and Rename Method: A combination of moving a method to another class and renaming it.
* Parameterize Test: Converts a standard test method into a parameterized test.
* Parameterize Variable: Replaces a hard-coded value within a method with a parameter.
* Pull Up Attribute: Moves a field from a subclass to its parent superclass.
* Pull Up Method: Moves a method from a subclass to its parent superclass.
* Push Down Attribute: Moves a field from a superclass to a specific subclass.
* Push Down Method: Moves a method from a superclass to a specific subclass.
* Remove Class Annotation: Deletes an annotation from a class declaration.
* Remove Class Modifier: Removes a modifier like static, final, or abstract from a class.
* Remove Attribute Annotation: Deletes an annotation from a field.
* Remove Attribute Modifier: Removes a modifier like static, final, or volatile from a field.
* Remove Method Annotation: Deletes an annotation from a method.
* Remove Method Modifier: Removes a modifier like static, final, or abstract from a method.
* Remove Parameter: Removes an existing parameter from a method's signature.
* Remove Parameter Annotation: Deletes an annotation from a parameter.
* Remove Thrown Exception Type: Removes an exception from a method's throws clause.
* Rename Attribute: Changes the name of a field.
* Rename Class: Changes the name of a class.
* Rename Method: Changes the name of a method.
* Rename Package: Changes the name of a package.
* Reorder Parameter: Changes the order of parameters in a method's signature.
* Split Attribute: Splits a single field into multiple fields.
* Split Package: Decomposes a large package into smaller packages.
* Split Parameter: Breaks a single parameter object into multiple parameters.


Medium-level (M) Refactorings (21 types)

These refactorings change both the signature and the code block.

* Change Attribute Type: Modifies the data type of a class field.
* Change Parameter Type: Modifies the data type of a method's parameter.
* Change Return Type: Modifies the data type of a method's return value.
* Extract Attribute: Moves an expression from a method's code block into a new class field.
* Extract Class: Creates a new class and moves some fields and methods from an existing class to it.
* Extract Method: Creates a new method from a fragment of code within an existing method.
* Extract Subclass: Creates a new subclass from an existing class.
* Extract and Move Method: A combination of extracting code into a new method and then moving that method.
* Inline Attribute: Replaces the usage of a field with its value and removes the field.
* Inline Method: Replaces a method call with the body of the called method and removes the original method.
* Localize Parameter: Replaces a parameter with a local variable, deriving its value from within the method.
* Merge Method: Combines the bodies of multiple methods into a single method.
* Move and Inline Method: A combination of moving a method and then inlining it at its new call site.
* Parameterize Attribute: Replaces the use of a hard-coded class attribute with a method parameter.
* Replace Anonymous with Class: Converts an anonymous inner class into a named class.
* Replace Attribute with Variable: Demotes a class field to a local variable.
* Replace Variable with Attribute: Promotes a local variable to a class field.
* Split Class: Decomposes a large class into two or more smaller classes.
* Split Method: Decomposes a long method into smaller, private helper methods.


Low-level (L) Refactorings (24 types)

These refactorings make changes confined to a code block.

* Add Parameter Modifier: Adds a modifier (like final) to a parameter.
* Add Variable Annotation: Adds an annotation to a local variable.
* Add Variable Modifier: Adds a modifier (like final) to a local variable.
* Assert Throws: Wraps code in a test to assert that a specific exception is thrown.
* Assert Timeout: Wraps a test case to assert that it completes within a specified time.
* Change Variable Type: Modifies the data type of a local variable.
* Extract Variable: Replaces a complex expression with a new, well-named local variable.
* Inline Variable: Replaces a local variable with the expression it contains.
* Invert Condition: Swaps the if and else blocks of a conditional statement by inverting the condition.
* Merge Catch: Combines multiple catch blocks with identical logic into a single block.
* Merge Conditional: Combines nested or sequential if statements into a single conditional block.
* Merge Variable: Combines two or more local variables into one.
* Modify Variable Annotation: Changes an existing annotation on a local variable.
* Move Code: Moves a block of statements from one method to another existing method.
* Remove Parameter Modifier: Removes a modifier from a parameter.
* Remove Variable Annotation: Deletes an annotation from a local variable.
* Remove Variable Modifier: Removes a modifier from a local variable.
* Rename Parameter: Changes the name of a method parameter within the method body for clarity.
* Rename Variable: Changes the name of a local variable to be more descriptive.
* Replace Anonymous with Lambda: Converts an anonymous inner class to a more concise lambda expression.
* Replace Attribute (with Attribute): Changes the usage of one field to another within a method body.
* Replace Conditional With Ternary: Replaces a simple if-else statement with a ternary operator (?:).
* Replace Generic With Diamond: Replaces explicit generic type arguments with the diamond operator (<>).
* Replace Loop with Pipeline: Replaces an imperative loop with a declarative stream or pipeline API call.
* Replace Pipeline with Loop: Replaces a stream or pipeline API call with a traditional imperative loop.
* Split Conditional: Decomposes a complex conditional block into simpler, separate checks.
* Split Variable: Splits a single local variable into multiple variables.
* Try With Resources: Converts a try-finally block to a try-with-resources statement.

