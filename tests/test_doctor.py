from insar_unwrap import doctor


def test_optional_missing_does_not_fail(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: None)
    results = doctor.run_checks()
    snaphu = [r for r in results if r.name == "snaphu"][0]
    assert not snaphu.ok and snaphu.severity == "optional" and snaphu.status == "MISSING"
    assert not doctor.has_required_failure(results)
    assert "Optional items missing" in doctor.format_report(results)


def test_required_failure_is_distinguished():
    broken = doctor.check_class("SQA", "dwave.samplers", "NoSuchSampler", "x")
    assert not broken.ok and broken.severity == "required" and broken.status == "FAIL"
    assert doctor.has_required_failure([broken])
    assert "not usable" in doctor.format_report([broken])


def test_sampler_classes_import():
    results = {r.name: r for r in doctor.run_checks()}
    assert results["SA sampler"].ok
    assert results["SQA sampler"].ok
    assert "PathIntegralAnnealingSampler" in results["SQA sampler"].detail
