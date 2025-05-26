import type {
	IAuthenticateGeneric,
	ICredentialTestRequest,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class NvidiaTritonApi implements ICredentialType {
	name = 'nvidiaTritonApi';
	displayName = 'NVIDIA Triton';
	documentationUrl = 'nvidiaTritonApi';

	properties: INodeProperties[] = [
		{
			displayName: 'Server URL',
			name: 'serverUrl',
			type: 'string',
			default: 'http://localhost:8000',
			required: false,
			description: 'Base URL of your NVIDIA Triton Inference Server (e.g. http://localhost:8000).',
		},
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			required: false,
			description: 'Optional API key for authentication. Leave blank if not required.',
		},
	];

	authenticate: IAuthenticateGeneric = {
		type: 'generic',
		properties: {
			headers: {
				Authorization: '=Bearer {{$credentials.apiKey}}',
			},
		},
	};

	test: ICredentialTestRequest = {
		request: {
			baseURL: '={{$credentials.serverUrl}}',
			url: '/v1/models',
			method: 'GET',
		},
	};
}
