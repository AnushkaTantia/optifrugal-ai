from streamlit.testing.v1 import AppTest
import itertools
def run(setup=None):
    at = AppTest.from_file("app.py", default_timeout=120).run()
    if setup: setup(at); at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at
at = run(); print("default ok | metrics:", [(m.label, m.value, m.delta) for m in at.metric])
for persona in ["Student / heavy screen use", "Child / active wearer", "Fashion-focused high-power", "Eco-conscious professional"]:
    def s(a, p=persona): a.sidebar.selectbox[0].set_value(p)
    at = run(s); print(persona, "ok:", [m.value for m in at.metric][:3])
for P in [-9.0, 0.0, 5.0]:
    def s(a, P=P): a.sidebar.slider[0].set_value(P)
    at = run(s); print("P", P, "ok", len(at.error), "errors-boxes;", [m.value for m in at.metric][:3])
