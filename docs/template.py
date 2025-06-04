"""
This is a short description of the module. You can, for instance, explain in few lines
that you have some classes that do stuff. The name of the class should be referenced
as ``AClass`` to typeset it correctly in the docs.
"""

class AClass:
    """
    This is the docstring of ``AClass``. You should give a short
    description of it. Now let's describe the attributes after a blank line:

    Attributes
    ----------
    foo: int
        The attribute foo of type int. Use the semicolon after the name without a space, and separate the type with a space.
        The description of the attribute is on a newline with 1 tab (4 spaces) indentation.
    """

    def __init__(self, foo: int):
        # The __init__ function should not have a docstring, nor the class methods should be described in the
        # class docstring. Notice there is no blank line after the last attribute but there is one before the
        # def __init__
        self.foo = foo

    def say_hello(self, name: str) -> str:
        """
        This is the docstring of the method. Let's describe the signature after a blank line:

        Parameters
        ----------
        name: str
            Same as for class attributes. No blank line for multiple parameters but one blank line before
            the description of what is returned.

        Returns
        -------
        str
            Only the type.
        """
        return f"Hello, {name}, my attribute is {self.foo}!"  # Here you can leave a blank line before or not
    

class AnotherClass:
    """
    Docs of this class. Here we have an attribute of type ``AClass``
    and the docstring is a little different.

    Attributes
    ----------
    elem: AClass
        Here the type is described directly after its name, but in this description you
        should use :class:`~root.package.module.AClass` to have a symbolic link.
        The same goes for function parameters. This should be used for classes belonging to
        this project and not for other classes from both Python or third-party APIs.
    """

    def __init__(self):
        self.elem = AClass(2)
