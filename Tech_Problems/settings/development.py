from .common import *

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = ')!lb4$ln(6_mkd)^qrf#xw%6^^gtbjo2syjlz)=)tkrt!d9@il'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Database
# https://docs.djangoproject.com/en/2.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}
