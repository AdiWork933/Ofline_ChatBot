from django.urls import path
from .views import (
    hello, ask_question, admin_upload_file, reload,
    generate_token, delete_file, list_uploaded_files, get_uploaded_files,
    # New imports for Super Admin functionality
    add_sub_admin, delete_sub_admin, list_sub_admins
)
urlpatterns = [
    # ── Chat / Q&A ───────────────────────────────────────────
    path('hello/', hello, name='hello'),
    path('ask/', ask_question, name='ask_question'),
    path('ask/status=<str:status_folder>/', ask_question, name='ask_question_by_status'),

    # ── Admin Authentication ──────────────────────────────────
    path('admin/login/', generate_token, name='generate_token'),

    # ── Super Admin User Management ───────────────────────────
    path('admin/sub-admins/add/', add_sub_admin, name='add_sub_admin'),
    path('admin/sub-admins/delete/', delete_sub_admin, name='delete_sub_admin'),
    path('admin/sub-admins/', list_sub_admins, name='list_sub_admins'),

    # ── Upload ──────────────────────────────────────────────
    path('admin/upload/', admin_upload_file, name='admin_upload'),
    path('admin/upload/<str:status_folder>/', admin_upload_file, name='admin_upload_with_folder'),

    # ── Maintenance ──────────────────────────────────────────
    path('admin/reload/', reload, name='reload'),
    
    # ── Delete ──────────────────────────────────────────────
    path('admin/delete/<str:status_folder>/', delete_file, name='delete_multiple_files'),
    path('admin/delete/<str:status_folder>/<str:filename>/', delete_file, name='delete_single_file'),

    # ── View / Download Files ───────────────────────────────
    # These are still publicly accessible (no @require_token applied)
    path('admin/view/', list_uploaded_files, name='view_all_files'),
    path('admin/view/<str:status_folder>/', list_uploaded_files, name='view_folder_files'),
    path('admin/download/<str:folder_name>/', get_uploaded_files, name='download_multiple_files'),
    path('admin/download/<str:folder_name>/<str:file_name>/', get_uploaded_files, name='download_single_file'),
]


