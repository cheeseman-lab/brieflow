"""Registration of user-defined per-cell phenotype features."""

import hashlib
import inspect
import textwrap
from numbers import Number

import numpy as np

CUSTOM_FEATURE_PREFIX = "custom"
CUSTOM_FEATURE_COMPARTMENTS = ("nucleus", "cell", "cytoplasm")
DEFAULT_CUSTOM_FEATURE_COMPARTMENT = "nucleus"


def custom_feature_column(
    name, source_hash, compartment=DEFAULT_CUSTOM_FEATURE_COMPARTMENT
):
    """Build the namespaced column name for a registered feature.

    Args:
        name (str): Name of the feature function.
        source_hash (str): Short hash of the feature source.
        compartment (str, optional): Compartment the feature is measured on. Default
            is "nucleus".

    Returns:
        str: Column name of the form {compartment}_custom_{name}_{source_hash}.
    """
    return f"{compartment}_{CUSTOM_FEATURE_PREFIX}_{name}_{source_hash}"


def register_custom_features(features):
    """Convert feature functions into config-serializable definitions.

    Each function is stored as its own source text so the workflow, which runs in a
    separate process from the notebook, can rebuild it, and so the definition behind
    a column is recoverable from the config alone. That only holds for a self-contained
    function, so one closing over a notebook name is rejected here rather than on a
    compute node hours later.

    Args:
        features (list): Feature functions taking a regionprops-like region and
            returning a single number, matching the foci_features convention. Each
            must define or import every name it uses. An entry may instead be a
            (function, compartment) pair, where compartment is one of "nucleus",
            "cell", or "cytoplasm"; a bare function is measured on the nucleus.

    Returns:
        list: Definitions with name, source, hash, compartment, and column keys.

    Raises:
        ValueError: If a function is anonymous, is registered twice on the same
            compartment, declares an unknown compartment, closes over names it does
            not define, or its source cannot be captured.
    """
    definitions = []
    registered = set()

    for entry in features:
        if isinstance(entry, (tuple, list)):
            if len(entry) != 2:
                raise ValueError(
                    "Custom features declared with a compartment must be a "
                    f"(function, compartment) pair, got {entry!r}"
                )
            func, compartment = entry
        else:
            func, compartment = entry, DEFAULT_CUSTOM_FEATURE_COMPARTMENT

        name = getattr(func, "__name__", None)
        if name is None or not name.isidentifier():
            raise ValueError(
                "Custom features must be named functions defined with def, "
                f"got {func!r}"
            )

        compartment = _validated_compartment(name, compartment)
        if (name, compartment) in registered:
            raise ValueError(
                f"Custom feature '{name}' is registered more than once on the "
                f"{compartment} compartment"
            )
        registered.add((name, compartment))

        try:
            source = textwrap.dedent(inspect.getsource(func))
        except (OSError, TypeError) as error:
            raise ValueError(
                f"Could not capture the source of custom feature '{name}'; define it "
                "with def in a notebook cell or module"
            ) from error

        closure = inspect.getclosurevars(func)
        closed_over = sorted(closure.globals) + sorted(closure.nonlocals)
        if closed_over:
            raise ValueError(
                f"Custom feature '{name}' uses names it does not define: {closed_over}. "
                "Only the source is carried to the workflow, so inline the values and "
                "move any imports inside the function"
            )

        source_hash = hashlib.sha256(source.encode()).hexdigest()[:8]
        definitions.append(
            {
                "name": name,
                "source": source,
                "hash": source_hash,
                "compartment": compartment,
                "column": custom_feature_column(name, source_hash, compartment),
            }
        )

    return definitions


def load_custom_features(definitions):
    """Rebuild registered features as per-compartment feature dictionaries for extraction.

    Sources are executed in an empty namespace, matching the self-contained definitions
    register_custom_features accepts.

    Args:
        definitions (list or None): Definitions produced by register_custom_features.

    Returns:
        dict: Mapping of compartment to a mapping of namespaced column name to feature
            function, in CUSTOM_FEATURE_COMPARTMENTS order.

    Raises:
        ValueError: If a definition declares an unknown compartment, or its source does
            not define its named function.
    """
    features = {}

    for definition in definitions or []:
        name = definition["name"]
        compartment = _validated_compartment(
            name, definition.get("compartment", DEFAULT_CUSTOM_FEATURE_COMPARTMENT)
        )
        namespace = {}
        exec(definition["source"], namespace)

        func = namespace.get(name)
        if not callable(func):
            raise ValueError(
                f"Source of custom feature '{name}' does not define a function "
                f"named '{name}'"
            )

        features.setdefault(compartment, {})[definition["column"]] = _checked_feature(
            name, func
        )

    return {
        compartment: features[compartment]
        for compartment in CUSTOM_FEATURE_COMPARTMENTS
        if compartment in features
    }


def _validated_compartment(name, compartment):
    """Check that a declared compartment is one the extraction measures on.

    Args:
        name (str): Name of the feature function.
        compartment (str): Declared compartment.

    Returns:
        str: The declared compartment.

    Raises:
        ValueError: If the compartment is not one of CUSTOM_FEATURE_COMPARTMENTS.
    """
    if compartment not in CUSTOM_FEATURE_COMPARTMENTS:
        raise ValueError(
            f"Custom feature '{name}' declares unknown compartment {compartment!r}, "
            f"expected one of {list(CUSTOM_FEATURE_COMPARTMENTS)}"
        )

    return compartment


def _checked_feature(name, func):
    """Wrap a feature function so a bad cell names the feature instead of emitting NaN.

    Args:
        name (str): Name of the feature function.
        func (callable): Feature function to wrap.

    Returns:
        callable: Feature function that raises on failure or a non-numeric result.
    """

    def checked(region):
        try:
            value = func(region)
        except Exception as error:
            raise ValueError(
                f"Custom feature '{name}' failed on label {region.label}: {error}"
            ) from error

        if not isinstance(value, (Number, np.bool_)):
            raise ValueError(
                f"Custom feature '{name}' returned {type(value).__name__} on label "
                f"{region.label}, expected a single number"
            )

        return value

    return checked
