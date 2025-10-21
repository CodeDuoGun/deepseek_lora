from dynaconf import Dynaconf

config = Dynaconf(
    envvar_prefix_for_dynaconf=False,
    load_dotenv=True,
    environments=True,
    settings_files=["app/config.yaml"],
)