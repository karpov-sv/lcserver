"""
URL configuration for lcserver project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, re_path, include
from django.conf import settings

from django.contrib.auth import views as auth_views

from . import views
from . import views_celery

urlpatterns = [
    path('', views.index, name='index'),

    # Targets
    path('targets/', views.targets, {'id':None}, name='targets'),
    path('targets/<int:id>', views.targets, name='targets'),

    path('targets/<int:id>/files/', views.target_files, {'path': ''}, name='target_files'),
    path('targets/<int:id>/files/<path:path>', views.target_files, name='target_files'),
    path('targets/<int:id>/preview/<path:path>', views.target_preview, name='target_preview'),
    path('targets/<int:id>/view/<path:path>', views.target_download, {'attachment': False}, name='target_view'),
    path('targets/<int:id>/download/<path:path>', views.target_download, {'attachment': True}, name='target_download'),

    # Celery queue
    path('queue/', views_celery.view_queue, {'id': None}, name='queue'),
    path('queue/<slug:id>', views_celery.view_queue, name='queue'),
    path('queue/<slug:id>/state', views_celery.get_queue, name='queue_state'),

    # Auth
    path('login/', auth_views.LoginView.as_view(), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # Admin panel
    path('admin/', admin.site.urls),
]
