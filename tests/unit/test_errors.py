from symlmm.errors import NotDecomposable, ParameterConstraintError, SymlmmError


def test_not_decomposable_carries_reason_and_suggestion():
    error = NotDecomposable("AR(1) has no exact split", "use InducedCovariance")
    assert error.reason == "AR(1) has no exact split"
    assert error.suggestion == "use InducedCovariance"
    assert isinstance(error, SymlmmError)


def test_parameter_constraint_error_carries_violations():
    error = ParameterConstraintError(["rho < 1"])
    assert error.violations == ("rho < 1",)
