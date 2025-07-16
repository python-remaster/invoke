import os
import pickle
import re
import sys
from unittest.mock import Mock, call, patch

from _util import _Dummy, mock_subprocess
from pytest import fixture, mark, raises, skip
from pytest_relaxed import trap

from invoke import (
    AuthFailure,
    Config,
    Context,
    FailingResponder,
    MockContext,
    ResponseNotAccepted,
    Result,
    StreamWatcher,
)

_escaped_prompt = re.escape(Config().sudo.prompt)


# Context_:
class init:
    "__init__"

    def test_takes_optional_config_arg(self, ctx):
        # Meh-tastic doesn't-barf tests. MEH.
        Context(config={"foo": "bar"})

class methods_exposed:
    def _expect_attr(self, attr):
        ctx = Context()
        assert hasattr(ctx, attr) and callable(getattr(ctx, attr))

    class run:
        # NOTE: actual behavior of command running is tested in runners.py
        def test_exists(self):
            self._expect_attr("run")

        def test_defaults_to_Local(self, local_path, ctx):
            ctx.run("foo")
            assert local_path.mock_calls == [call(ctx), call().run("foo")]

        def test_honors_runner_config_setting(self, ctx):
            runner_class = Mock()
            ctx.config = Config({"runners": {"local": runner_class}})
            ctx.run("foo")
            assert runner_class.mock_calls == [call(ctx), call().run("foo")]

    def test_sudo(self):
        self._expect_attr("sudo")

class ConfigProxy:
    "Dict-like proxy for self.config"

    def setup_method(self):
        self.ctx = Context(
            config=Config(defaults={"foo": "bar", "biz": {"baz": "boz"}})
        )

    def test_direct_access_allowed(self):
        assert self.ctx.config.__class__ == Config
        assert self.ctx.config["foo"] == "bar"
        assert self.ctx.config.foo == "bar"

    def test_config_attr_may_be_overwritten_at_runtime(self):
        new_config = Config(defaults={"foo": "notbar"})
        self.ctx.config = new_config
        assert self.ctx.foo == "notbar"

    def test_getitem(self):
        "___getitem__"
        assert self.ctx["foo"] == "bar"
        assert self.ctx["biz"]["baz"] == "boz"

    def test_getattr(self):
        "__getattr__"
        assert self.ctx.foo == "bar"
        assert self.ctx.biz.baz == "boz"

    def test_get(self):
        assert self.ctx.get("foo") == "bar"
        assert self.ctx.get("nope", "wut") == "wut"
        assert self.ctx.biz.get("nope", "hrm") == "hrm"

    def test_pop(self):
        assert self.ctx.pop("foo") == "bar"
        assert self.ctx.pop("foo", "notbar") == "notbar"
        assert self.ctx.biz.pop("baz") == "boz"

    def test_popitem(self):
        assert self.ctx.biz.popitem() == ("baz", "boz")
        del self.ctx["biz"]
        assert self.ctx.popitem() == ("foo", "bar")
        assert self.ctx.config == {}

    def test_del_(self):
        "del"
        del self.ctx["foo"]
        del self.ctx["biz"]["baz"]
        assert self.ctx.biz == {}
        del self.ctx["biz"]
        assert self.ctx.config == {}

    def test_clear(self):
        self.ctx.biz.clear()
        assert self.ctx.biz == {}
        self.ctx.clear()
        assert self.ctx.config == {}

    def test_setdefault(self):
        assert self.ctx.setdefault("foo") == "bar"
        assert self.ctx.biz.setdefault("baz") == "boz"
        assert self.ctx.setdefault("notfoo", "notbar") == "notbar"
        assert self.ctx.notfoo == "notbar"
        assert self.ctx.biz.setdefault("otherbaz", "otherboz") == "otherboz"
        assert self.ctx.biz.otherbaz == "otherboz"

    def test_update(self):
        self.ctx.update({"newkey": "newval"})
        assert self.ctx["newkey"] == "newval"
        assert self.ctx.foo == "bar"
        self.ctx.biz.update(otherbaz="otherboz")
        assert self.ctx.biz.otherbaz == "otherboz"

# cwd:
def test_simple(ctx):
    ctx.command_cwds = ["a", "b"]
    assert ctx.cwd == os.path.join("a", "b")

def test_nested_absolute_path(ctx):
    ctx.command_cwds = ["a", "/b", "c"]
    assert ctx.cwd == os.path.join("/b", "c")

def test_multiple_absolute_paths(ctx):
    ctx.command_cwds = ["a", "/b", "c", "/d", "e"]
    assert ctx.cwd == os.path.join("/d", "e")

def test_home(ctx):
    ctx.command_cwds = ["a", "~b", "c"]
    assert ctx.cwd == os.path.join("~b", "c")


class cd:
    def test_should_apply_to_run(self, local_path, ctx):
        with ctx.cd("foo"):
            ctx.run("whoami")

        runner = local_path.return_value
        assert runner.run.called, "run() never called runner.run()!"
        assert runner.run.call_args[0][0] == "cd foo && whoami"

    def test_should_apply_to_sudo(self, local_path, ctx):
        runner = local_path.return_value
        with ctx.cd("foo"):
            ctx.sudo("whoami")

        cmd = "sudo -S -p '[sudo] password: ' cd foo && whoami"
        assert runner.run.called, "sudo() never called runner.run()!"
        assert runner.run.call_args[0][0] == cmd

    def test_should_occur_before_prefixes(self, local_path, ctx):
        runner = local_path.return_value
        with ctx.prefix("source venv"):
            with ctx.cd("foo"):
                ctx.run("whoami")

        cmd = "cd foo && source venv && whoami"
        assert runner.run.called, "run() never called runner.run()!"
        assert runner.run.call_args[0][0] == cmd

    def test_should_use_finally_to_revert_changes_on_exceptions(
        self, local_path, ctx
    ):
        class Oops(Exception):
            pass

        runner = local_path.return_value
        try:
            with ctx.cd("foo"):
                ctx.run("whoami")
                assert runner.run.call_args[0][0] == "cd foo && whoami"
                raise Oops
        except Oops:
            pass
        ctx.run("ls")
        # When bug present, this would be "cd foo && ls"
        assert runner.run.call_args[0][0] == "ls"

    def test_cd_should_accept_any_stringable_object(self, local_path, ctx):
        class Path:
            def __init__(self, value):
                self.value = value

            def __fspath__(self):
                return self.value

        runner = local_path.return_value

        with ctx.cd(Path("foo")):
            ctx.run("whoami")

        cmd = "cd foo && whoami"
        assert runner.run.call_args[0][0] == cmd


def test_prefixes_should_apply_to_run(local_path, ctx):
    runner = local_path.return_value
    with ctx.prefix("cd foo"):
        ctx.run("whoami")

    cmd = "cd foo && whoami"
    assert runner.run.called, "run() never called runner.run()!"
    assert runner.run.call_args[0][0] == cmd


def test_prefixes_should_apply_to_sudo(local_path, ctx):
    runner = local_path.return_value
    with ctx.prefix("cd foo"):
        ctx.sudo("whoami")

    cmd = "sudo -S -p '[sudo] password: ' cd foo && whoami"
    assert runner.run.called, "sudo() never called runner.run()!"
    assert runner.run.call_args[0][0] == cmd


def test_nesting_should_retain_order(local_path, ctx):
    runner = local_path.return_value
    with ctx.prefix("cd foo"):
        with ctx.prefix("cd bar"):
            ctx.run("whoami")
            cmd = "cd foo && cd bar && whoami"
            assert (
                runner.run.called
            ), "run() never called runner.run()!"  # noqa
            assert runner.run.call_args[0][0] == cmd

        ctx.run("whoami")
        cmd = "cd foo && whoami"
        assert runner.run.called, "run() never called runner.run()!"
        assert runner.run.call_args[0][0] == cmd

    # also test that prefixes do not persist
    cmd = "whoami"
    ctx.run("whoami")
    assert runner.run.called, "run() never called runner.run()!"
    assert runner.run.call_args[0][0] == cmd


def test_should_use_finally_to_revert_changes_on_exceptions(local_path, ctx):
    class Oops(Exception):
        pass

    runner = local_path.return_value
    try:
        with ctx.prefix("cd foo"):
            ctx.run("whoami")
            assert runner.run.call_args[0][0] == "cd foo && whoami"
            raise Oops
    except Oops:
        pass
    ctx.run("ls")
    # When bug present, this would be "cd foo && ls"
    assert runner.run.call_args[0][0] == "ls"


# sudo
def test_prefixes_command_with_sudo(local_path, ctx):
    runner = local_path.return_value
    ctx.sudo("whoami")
    # NOTE: implicitly tests default sudo.prompt conf value
    cmd = "sudo -S -p '[sudo] password: ' whoami"
    assert runner.run.called, "sudo() never called runner.run()!"
    assert runner.run.call_args[0][0] == cmd


def test_optional_user_argument_adds_u_and_H_flags(local_path, ctx):
    runner = local_path.return_value
    ctx.sudo("whoami", user="rando")
    cmd = "sudo -S -p '[sudo] password: ' -H -u rando whoami"
    assert runner.run.called, "sudo() never called runner.run()!"
    assert runner.run.call_args[0][0] == cmd


def test_honors_config_for_user_value(local_path, ctx):
    runner = local_path.return_value
    ctx.config = Config(overrides={"sudo": {"user": "rando"}})
    ctx.sudo("whoami")
    cmd = "sudo -S -p '[sudo] password: ' -H -u rando whoami"
    assert runner.run.call_args[0][0] == cmd


def test_user_kwarg_wins_over_config(local_path, ctx):
    runner = local_path.return_value
    ctx.config = Config(overrides={"sudo": {"user": "rando"}})
    ctx.sudo("whoami", user="calrissian")
    cmd = "sudo -S -p '[sudo] password: ' -H -u calrissian whoami"
    assert runner.run.call_args[0][0] == cmd


@trap
@mock_subprocess()
def test_echo_hides_extra_sudo_flags():
    skip()  # see TODO in sudo() re: clean output display
    ctx.config = Config(overrides={"runner": _Dummy})
    ctx.sudo("nope", echo=True)
    output = sys.stdout.getvalue()
    sys.__stderr__.write(repr(output) + "\n")
    assert "-S" not in output
    assert Context().sudo.prompt not in output
    assert "sudo nope" in output


def test_honors_config_for_prompt_value(local_path, ctx):
    runner = local_path.return_value
    ctx.config = Config(overrides={"sudo": {"prompt": "FEED ME: "}})
    ctx.sudo("whoami")
    cmd = "sudo -S -p 'FEED ME: ' whoami"
    assert runner.run.call_args[0][0] == cmd


def prompt_value_is_properly_shell_escaped():
    # I.e. setting it to "here's johnny!" doesn't explode.
    # NOTE: possibly best to tie into issue #2
    skip()


def test_explicit_env_vars_are_preserved(local_path, ctx):
    runner = local_path.return_value
    ctx.sudo(
        "whoami",
        env={"GRATUITOUS_ENVIRONMENT_VARIABLE": "arbitrary value"},
    )
    assert (
        "--preserve-env='GRATUITOUS_ENVIRONMENT_VARIABLE'"
        in runner.run.call_args[0][0]
    )

class sudo:
    def _expect_responses(self, expected, config=None, kwargs=None):
        """
        Execute mocked sudo(), expecting watchers= kwarg in its run().

        * expected: list of 2-tuples of FailingResponder prompt/response
        * config: Config object, if an overridden one is needed
        * kwargs: sudo() kwargs, if needed
        """
        if kwargs is None:
            kwargs = {}
        Local = Mock()
        runner = Local.return_value
        context = Context(config=config) if config else Context()
        context.config.runners.local = Local
        context.sudo("whoami", **kwargs)
        # Tease out the interesting bits - pattern/response - ignoring the
        # sentinel, etc for now.
        prompt_responses = [
            (watcher.pattern, watcher.response)
            for watcher in runner.run.call_args[1]["watchers"]
        ]
        assert prompt_responses == expected

    def test_autoresponds_with_password_kwarg(self):
        # NOTE: technically duplicates the unitty test(s) in watcher tests.
        expected = [(_escaped_prompt, "secret\n")]
        self._expect_responses(expected, kwargs={"password": "secret"})

    def test_honors_configured_sudo_password(self):
        config = Config(overrides={"sudo": {"password": "secret"}})
        expected = [(_escaped_prompt, "secret\n")]
        self._expect_responses(expected, config=config)

    def test_sudo_password_kwarg_wins_over_config(self):
        config = Config(overrides={"sudo": {"password": "notsecret"}})
        kwargs = {"password": "secret"}
        expected = [(_escaped_prompt, "secret\n")]
        self._expect_responses(expected, config=config, kwargs=kwargs)

    class auto_response_merges_with_other_responses:
        def setup_method(self):
            class DummyWatcher(StreamWatcher):
                def submit(self, stream):
                    pass

            self.watcher_klass = DummyWatcher

        def test_kwarg_only_adds_to_kwarg(self, local_path):
            runner = local_path.return_value
            context = Context()
            watcher = self.watcher_klass()
            context.sudo("whoami", watchers=[watcher])
            # When sudo() called w/ user-specified watchers, we add ours to
            # that list
            watchers = runner.run.call_args[1]["watchers"]
            # Will raise ValueError if not in the list
            watchers.remove(watcher)
            # Only remaining item in list should be our sudo responder
            assert len(watchers) == 1
            assert isinstance(watchers[0], FailingResponder)
            assert watchers[0].pattern == _escaped_prompt

        def test_config_only(self, local_path, ctx):
            runner = local_path.return_value
            # Set a config-driven list of watchers
            watcher = self.watcher_klass()
            overrides = {"run": {"watchers": [watcher]}}
            ctx.config = Config(overrides=overrides)
            ctx.sudo("whoami")
            # Expect that sudo() extracted that config value & put it into
            # the kwarg level. (See comment in sudo() about why...)
            watchers = runner.run.call_args[1]["watchers"]
            # Will raise ValueError if not in the list
            watchers.remove(watcher)
            # Only remaining item in list should be our sudo responder
            assert len(watchers) == 1
            assert isinstance(watchers[0], FailingResponder)
            assert watchers[0].pattern == _escaped_prompt

        def test_config_use_does_not_modify_config(self, local_path):
            runner = local_path.return_value
            watcher = self.watcher_klass()
            overrides = {"run": {"watchers": [watcher]}}
            config = Config(overrides=overrides)
            Context(config=config).sudo("whoami")
            # Here, 'watchers' is _the same object_ as was passed into
            # run(watchers=...).
            watchers = runner.run.call_args[1]["watchers"]
            # We want to make sure that what's in the config we just
            # generated, is untouched by the manipulation done inside
            # sudo().
            # First, that they aren't the same obj
            err = "Found sudo() reusing config watchers list directly!"
            assert watchers is not config.run.watchers, err
            # And that the list is as it was before (i.e. it is not both
            # our watcher and the sudo()-added one)
            err = "Our config watchers list was modified!"
            assert config.run.watchers == [watcher], err

        def test_both_kwarg_and_config(self, local_path):
            runner = local_path.return_value
            # Set a config-driven list of watchers
            conf_watcher = self.watcher_klass()
            overrides = {"run": {"watchers": [conf_watcher]}}
            config = Config(overrides=overrides)
            # AND supply a DIFFERENT kwarg-driven list of watchers
            kwarg_watcher = self.watcher_klass()
            Context(config=config).sudo("whoami", watchers=[kwarg_watcher])
            # Expect that the kwarg watcher and our internal one were the
            # final result.
            watchers = runner.run.call_args[1]["watchers"]
            # Will raise ValueError if not in the list. .remove() uses
            # identity testing, so two instances of self.watcher_klass will
            # be different values here.
            watchers.remove(kwarg_watcher)
            # Only remaining item in list should be our sudo responder
            assert len(watchers) == 1
            assert conf_watcher not in watchers  # Extra sanity
            assert isinstance(watchers[0], FailingResponder)
            assert watchers[0].pattern == _escaped_prompt

    def test_passes_through_other_run_kwargs(self, local_path):
        runner = local_path.return_value
        Context().sudo(
            "whoami", echo=True, warn=False, hide=True, encoding="ascii"
        )
        assert runner.run.called, "sudo() never called runner.run()!"
        kwargs = runner.run.call_args[1]
        assert kwargs["echo"] is True
        assert kwargs["warn"] is False
        assert kwargs["hide"] is True
        assert kwargs["encoding"] == "ascii"

    def test_returns_run_result(self, local_path):
        runner = local_path.return_value
        expected = runner.run.return_value
        result = Context().sudo("whoami")
        err = "sudo() did not return run()'s return value!"
        assert result is expected, err

    @mock_subprocess(out="something", exit=None)
    def test_raises_auth_failure_when_failure_detected(self):
        with patch("invoke.context.FailingResponder") as klass:
            unacceptable = Mock(side_effect=ResponseNotAccepted)
            klass.return_value.submit = unacceptable
            excepted = False
            try:
                config = Config(overrides={"sudo": {"password": "nope"}})
                Context(config=config).sudo("meh", hide=True)
            except AuthFailure as e:
                # Basic sanity checks; most of this is really tested in
                # Runner tests.
                assert e.result.exited is None
                expected = "The password submitted to prompt '[sudo] password: ' was rejected."  # noqa
                assert str(e) == expected
                excepted = True
            # Can't use except/else as that masks other real exceptions,
            # such as incorrectly unhandled ThreadErrors
            if not excepted:
                assert False, "Did not raise AuthFailure!"


def test_can_be_pickled(ctx):
    ctx.foo = {"bar": {"biz": ["baz", "buzz"]}}
    c2 = pickle.loads(pickle.dumps(ctx))
    assert ctx == c2
    assert ctx is not c2
    assert ctx.foo.bar.biz is not c2.foo.bar.biz


def test_init_still_acts_like_superclass_init():
    # No required args
    assert isinstance(MockContext().config, Config)
    config = Config(overrides={"foo": "bar"})
    # Posarg
    assert MockContext(config).config is config
    # Kwarg
    assert MockContext(config=config).config is config


def test_non_config_init_kwargs_used_as_return_values_for_methods():
    ctx = MockContext(run=Result("some output"))
    assert ctx.run("doesn't mattress").stdout == "some output"


def test_return_value_kwargs_can_take_iterables_too():
    ctx = MockContext(run=(Result("some output"), Result("more!")))
    assert ctx.run("doesn't mattress").stdout == "some output"
    assert ctx.run("still doesn't mattress").stdout == "more!"


def test_return_value_kwargs_may_be_command_string_maps():
    ctx = MockContext(run={"foo": Result("bar")})
    assert ctx.run("foo").stdout == "bar"


def test_return_value_map_kwargs_may_take_iterables_too():
    ctx = MockContext(run={"foo": (Result("bar"), Result("biz"))})
    assert ctx.run("foo").stdout == "bar"
    assert ctx.run("foo").stdout == "biz"


def test_regexen_return_value_map_keys_match_on_command():
    ctx = MockContext(
        run={"string": Result("yup"), re.compile(r"foo.*"): Result("bar")}
    )
    assert ctx.run("string").stdout == "yup"
    assert ctx.run("foobar").stdout == "bar"


# MockContext
# boolean_result_shorthand:
def test_as_singleton_args():
    assert MockContext(run=True).run("anything").ok
    assert not MockContext(run=False).run("anything", warn=True).ok


def test_as_iterables():
    ctx = MockContext(run=[True, False])
    assert ctx.run("anything").ok
    assert not ctx.run("anything", warn=True).ok


def test_as_dict_values():
    ctx = MockContext(run=dict(foo=True, bar=False))
    assert ctx.run("foo").ok
    assert not ctx.run("bar", warn=True).ok


def test_as_singleton_args():
    assert MockContext(run="foo").run("anything").stdout == "foo"


def test_as_iterables():
    ctx = MockContext(run=["definition", "of", "insanity"])
    assert ctx.run("anything").stdout == "definition"
    assert ctx.run("anything").stdout == "of"
    assert ctx.run("anything").stdout == "insanity"


def test_as_dict_values():
    ctx = MockContext(run=dict(foo="foo", bar="bar"))
    assert ctx.run("foo").stdout == "foo"
    assert ctx.run("bar").stdout == "bar"


# commands_injected_into_Result:
@mark.parametrize("kwargs", ({}, {"command": ""}, {"command": None}))
def test_when_not_set_or_falsey(kwargs):
    c = MockContext(run={"foo": Result("bar", **kwargs)})
    assert c.run("foo").command == "foo"


def test_does_not_occur_when_truthy():
    # Not sure why you'd want this but whatevs!
    c = MockContext(run={"foo": Result("bar", command="nope")})
    assert c.run("foo").command == "nope"  # not "bar"


def methods_with_no_kwarg_values_raise_NotImplementedError():
    with raises(NotImplementedError):
        MockContext().run("onoz I did not anticipate this would happen")


def test_does_not_consume_results_by_default():
    ctx = MockContext(
        run=dict(
            singleton=True,  # will repeat
            wassup=Result("yo"),  # ditto
            iterable=[Result("tick"), Result("tock")],  # will not
        ),
    )
    assert ctx.run("singleton").ok
    assert ctx.run("singleton").ok  # not consumed
    assert ctx.run("wassup").ok
    assert ctx.run("wassup").ok  # not consumed
    assert ctx.run("iterable").stdout == "tick"
    assert ctx.run("iterable").stdout == "tock"
    assert ctx.run("iterable").stdout == "tick"  # not consumed
    assert ctx.run("iterable").stdout == "tock"


def test_consumes_singleton_results_when_repeat_False():
    ctx = MockContext(
        repeat=False,
        run=dict(
            singleton=True,
            wassup=Result("yo"),
            iterable=[Result("tick"), Result("tock")],
        ),
    )
    assert ctx.run("singleton").ok
    with raises(NotImplementedError):  # was consumed
        ctx.run("singleton")
    assert ctx.run("wassup").ok
    with raises(NotImplementedError):  # was consumed
        ctx.run("wassup")
    assert ctx.run("iterable").stdout == "tick"
    assert ctx.run("iterable").stdout == "tock"
    with raises(NotImplementedError):  # was consumed
        assert ctx.run("iterable")


def test_sudo_also_covered():
    c = MockContext(sudo=Result(stderr="super duper"))
    assert c.sudo("doesn't mattress").stderr == "super duper"
    try:
        MockContext().sudo("meh")
    except NotImplementedError as e:
        assert str(e) == "meh"
    else:
        assert False, "Did not get a NotImplementedError for sudo!"


class exhausted_nonrepeating_return_values_also_raise_NotImplementedError:
    def _expect_NotImplementedError(self, context):
        context.run("something")
        try:
            context.run("something")
        except NotImplementedError as e:
            assert str(e) == "something"
        else:
            assert False, "Didn't raise NotImplementedError"

    def test_single_value(self):
        self._expect_NotImplementedError(
            MockContext(run=Result("meh"), repeat=False)
        )

    def test_iterable(self):
        self._expect_NotImplementedError(
            MockContext(run=[Result("meh")], repeat=False)
        )

    def test_mapping_to_single_value(self):
        self._expect_NotImplementedError(
            MockContext(run={"something": Result("meh")}, repeat=False)
        )

    def test_mapping_to_iterable(self):
        self._expect_NotImplementedError(
            MockContext(run={"something": [Result("meh")]}, repeat=False)
        )


def unexpected_kwarg_type_yields_TypeError():
    with raises(TypeError):
        MockContext(run=123)


def test_run():
    ctx = MockContext(run={"foo": Result("bar")})
    assert ctx.run("foo").stdout == "bar"
    ctx.set_result_for("run", "foo", Result("biz"))
    assert ctx.run("foo").stdout == "biz"


@mark.parametrize(
    "kwargs", ({}, {"run": Result("foo")}, {"run": [Result("foo")]})
)
def test_run_results(kwargs):
    """Test can modify return value maps after instantiation and non-dict type
    instantiation values yield TypeErrors.
    """
    ctx = MockContext(**kwargs)
    with raises(TypeError):
        ctx.set_result_for("run", "whatever", Result("bar"))


def test_sudo():
    ctx = MockContext(sudo={"foo": Result("bar")})
    assert ctx.sudo("foo").stdout == "bar"
    ctx.set_result_for("sudo", "foo", Result("biz"))
    assert ctx.sudo("foo").stdout == "biz"


@mark.parametrize(
    "kwargs", ({}, {"run": Result("foo")}, {"run": [Result("foo")]})
)
def test_sudo(kwargs):
    """Test can modify return value maps after instantiation and non-dict type
    instantiation values yield TypeErrors.
    """
    ctx = MockContext(**kwargs)
    with raises(TypeError):
        ctx.set_result_for("sudo", "whatever", Result("bar"))


def test_wraps_run_and_sudo_with_Mock(clean_sys_modules):
    sys.modules["mock"] = None  # legacy
    sys.modules["unittest.mock"] = Mock(Mock=Mock)  # buffalo buffalo
    ctx = MockContext(
        run={"foo": Result("bar")}, sudo={"foo": Result("bar")}
    )
    assert isinstance(ctx.run, Mock)
    assert isinstance(ctx.sudo, Mock)
