# -*- coding: utf-8 -*- #
# Copyright 2016 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A command to install Application Default Credentials using a user account."""

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import textwrap

from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.command_lib.auth import auth_util as command_auth_util
from googlecloudsdk.command_lib.auth import flags
from googlecloudsdk.command_lib.auth import workforce_login_config as workforce_login_config_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import creds as creds_module
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.util import files


@base.UniverseCompatible
class Login(base.Command):
  r"""Acquire new user credentials to use for Application Default Credentials.

  Obtains user access credentials via a web flow and puts them in the
  well-known location for Application Default Credentials (ADC).

  This command is useful when you are developing code that would normally
  use a service account but need to run the code in a local development
  environment where it's easier to provide user credentials. The credentials
  will apply to all API calls that make use of the Application Default
  Credentials client library. Do not set the `GOOGLE_APPLICATION_CREDENTIALS`
  environment variable if you want to use the credentials generated by this
  command in your local development. This command tries to find a quota
  project from gcloud's context and write it to ADC so that Google client
  libraries can use it for billing and quota. Alternatively, you can use
  the `--client-id-file` flag. In this case, the project owning the client ID
  will be used for billing and quota. You can create the client ID file
  at https://console.cloud.google.com/apis/credentials.

  This command has no effect on the user account(s) set up by the
  `gcloud auth login` command.

  Any credentials previously generated by
  `gcloud auth application-default login` will be overwritten.
  """
  detailed_help = {
      'EXAMPLES':
          """\
          If you want your local application to temporarily use your own user
          credentials for API access, run:

            $ {command}

          If you'd like to login by passing in a file containing your own client
          id, run:

            $ {command} --client-id-file=clientid.json
          """
  }

  @staticmethod
  def Args(parser):
    """Set args for gcloud auth application-default login."""
    parser.add_argument(
        '--client-id-file',
        help='A file containing your own client id to use to login. If '
        '--client-id-file is specified, the quota project will not be '
        'written to ADC.')
    parser.add_argument(
        '--scopes',
        type=arg_parsers.ArgList(min_length=1),
        metavar='SCOPE',
        help='The names of the scopes to authorize for. By default '
        '{0} scopes are used. '
        'The list of possible scopes can be found at: '
        '[](https://developers.google.com/identity/protocols/googlescopes). '
        'To add scopes for applications outside of Google Cloud Platform, '
        'such as Google Drive, [create an OAuth Client ID]'
        '(https://support.google.com/cloud/answer/6158849) and provide it by '
        'using the --client-id-file flag. '
        .format(', '.join(auth_util.DEFAULT_SCOPES)))
    parser.add_argument(
        '--login-config',
        help='Path to the login configuration file (workforce pool, '
        'generated by the Cloud Console or '
        '`gcloud iam workforce-pools create-login-config`)',
        action=actions.StoreProperty(properties.VALUES.auth.login_config_file))
    parser.add_argument(
        'account',
        nargs='?',
        help=(
            'User account used for authorization. When the account specified'
            ' has valid credentials in the local credential store these'
            ' credentials will be re-used. Otherwise new ones will be fetched'
            ' and replace any stored credential.'
            ' This caching behavior is only available for user credentials.'
        ),
    )
    flags.AddQuotaProjectFlags(parser)
    flags.AddRemoteLoginFlags(parser, for_adc=True)

    parser.display_info.AddFormat('none')

  def Run(self, args):
    """Run the authentication command."""
    # TODO(b/203102970): Remove this condition check after the bug is resolved
    if properties.VALUES.auth.access_token_file.Get():
      raise c_store.FlowError(
          'auth/access_token_file or --access-token-file was set which is not '
          'compatible with this command. Please unset the property and rerun '
          'this command.'
      )

    if c_gce.Metadata().connected:
      message = textwrap.dedent("""
          You are running on a Google Compute Engine virtual machine.
          The service credentials associated with this virtual machine
          will automatically be used by Application Default
          Credentials, so it is not necessary to use this command.

          If you decide to proceed anyway, your user credentials may be visible
          to others with access to this virtual machine. Are you sure you want
          to authenticate with your personal account?
          """)
      console_io.PromptContinue(
          message=message, throw_if_unattended=True, cancel_on_no=True)

    command_auth_util.PromptIfADCEnvVarIsSet()
    if args.client_id_file and not args.launch_browser:
      raise c_exc.InvalidArgumentException(
          '--no-launch-browser',
          '`--no-launch-browser` flow no longer works with the '
          '`--client-id-file`. Please replace `--no-launch-browser` with '
          '`--no-browser`.'
      )

    # Currently the original scopes are not stored in the ADC. If they were, a
    # change in scopes could also be used to determine a cache hit.
    if args.account and not args.scopes:
      if ShouldUseCachedCredentials(args.account):
        log.warning(
            '\nValid credentials already exist for {}. To force refresh the'
            ' existing credentials, omit the "ACCOUNT" positional field.'
            .format(args.account)
        )
        return

    scopes = args.scopes or auth_util.DEFAULT_SCOPES
    flow_params = dict(
        no_launch_browser=not args.launch_browser,
        no_browser=not args.browser,
        remote_bootstrap=args.remote_bootstrap,
    )

    # 1. Try the 3PI web flow with --no-browser:
    #    This could be a 3PI flow initiated via --no-browser.
    #    If provider_name is present, then this is the 3PI flow.
    #    We can start the flow as is as the remote_bootstrap value will be used.
    if args.remote_bootstrap and 'provider_name' in args.remote_bootstrap:
      auth_util.DoInstalledAppBrowserFlowGoogleAuth(
          config.CLOUDSDK_EXTERNAL_ACCOUNT_SCOPES,
          auth_proxy_redirect_uri=(
              'https://sdk.cloud.google/applicationdefaultauthcode.html'
          ),
          **flow_params
      )
      return

    # 2. Try the 3PI web flow with a login configuration file.
    login_config_file = workforce_login_config_util.GetWorkforceLoginConfig()
    if login_config_file:
      if args.client_id_file:
        raise c_exc.ConflictingArgumentsException(
            '--client-id-file is not currently supported for third party login '
            'flows. ')
      if args.scopes:
        raise c_exc.ConflictingArgumentsException(
            '--scopes is not currently supported for third party login flows.')
      creds = workforce_login_config_util.DoWorkforceHeadfulLogin(
          login_config_file,
          True,
          **flow_params
      )
    else:
      # 3. Try the 1P web flow.
      properties.VALUES.auth.client_id.Set(
          auth_util.DEFAULT_CREDENTIALS_DEFAULT_CLIENT_ID)
      properties.VALUES.auth.client_secret.Set(
          auth_util.DEFAULT_CREDENTIALS_DEFAULT_CLIENT_SECRET)
      if auth_util.CLOUD_PLATFORM_SCOPE not in scopes:
        raise c_exc.InvalidArgumentException(
            '--scopes',
            '{} scope is required but not requested. Please include it in the'
            ' --scopes flag.'.format(auth_util.CLOUD_PLATFORM_SCOPE),
        )
      creds = auth_util.DoInstalledAppBrowserFlowGoogleAuth(
          scopes,
          client_id_file=args.client_id_file,
          auth_proxy_redirect_uri=(
              'https://sdk.cloud.google.com/applicationdefaultauthcode.html'
          ),
          **flow_params
      )
    if not creds:
      return

    if args.account and hasattr(creds, 'with_account'):
      _ = command_auth_util.ExtractAndValidateAccount(args.account, creds)
      creds = creds.with_account(args.account)

    target_impersonation_principal, delegates = None, None
    impersonation_service_accounts = (
        properties.VALUES.auth.impersonate_service_account.Get()
    )
    if impersonation_service_accounts:
      (target_impersonation_principal, delegates
      ) = c_store.ParseImpersonationAccounts(impersonation_service_accounts)
    if not target_impersonation_principal:
      if args.IsSpecified('client_id_file'):
        command_auth_util.DumpADC(creds, quota_project_disabled=False)
      elif args.disable_quota_project:
        command_auth_util.DumpADC(creds, quota_project_disabled=True)
      else:
        command_auth_util.DumpADCOptionalQuotaProject(creds)
    else:
      # TODO(b/184049366): Supports quota project with impersonated creds.
      command_auth_util.DumpImpersonatedServiceAccountToADC(
          creds,
          target_principal=target_impersonation_principal,
          delegates=delegates)
    return creds


def ShouldUseCachedCredentials(account):
  """Skip login if the existing ADC was provisioned for the requested account."""
  try:
    file_path = config.ADCFilePath()
    data = files.ReadFileContents(file_path)
  except files.Error:
    return False

  try:
    credential = creds_module.FromJsonGoogleAuth(data)
  except creds_module.UnknownCredentialsType:
    return False
  except creds_module.InvalidCredentialsError:
    return False

  cred_type = creds_module.CredentialTypeGoogleAuth.FromCredentials(credential)
  if cred_type == creds_module.CredentialTypeGoogleAuth.USER_ACCOUNT:
    if credential.account != account:
      return False

  return True
