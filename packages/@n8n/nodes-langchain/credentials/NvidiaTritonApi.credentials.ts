import type {
	IAuthenticateGeneric,
	ICredentialTestRequest,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class NvidiaTritonApi implements ICredentialType {
	name = 'nvidiaTritonApi';
	displayName = 'NVIDIA Triton API';
	documentationUrl = 'https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html'; // [[2]] (concept of documentationUrl)

	properties: INodeProperties[] = [ // [[5]] (structure of properties)
		{
			displayName: 'Triton Server URL',
			name: 'tritonServerUrl',
			type: 'string',
			required: true,
			default: 'http://localhost:8000',
			description: 'The base URL of your NVIDIA Triton Inference Server (e.g., http://localhost:8000 or https://your-triton-server.com/triton)',
			placeholder: 'http://localhost:8000',
		},
		{
			displayName: 'API Key (Optional)',
			name: 'apiKey',
			type: 'string',
			typeOptions: { password: true }, // [[5]] (example of password typeOption)
			default: '',
			description: 'Optional API Key or Bearer token for Triton if authentication is enabled on the server. This will be sent as a Bearer token.',
		},
	];

	authenticate: IAuthenticateGeneric = { // [[5]] (structure of authenticate)
		type: 'generic',
		properties: {
			headers: {
				// This header will be added if apiKey has a value.
				// If apiKey is empty, n8n's behavior is typically to send 'Authorization: Bearer '.
				// Triton should ideally ignore this for unauthenticated endpoints or if auth is not configured.
				Authorization: '=Bearer {{$credentials.apiKey}}',
			},
		},
	};

	test: ICredentialTestRequest = { // [[5]] (structure of test)
		request: {
			// Use the tritonServerUrl from the credentials as the base URL for the test request.
			baseURL: 'http://localhost:8000',
			// The /v2/health/live endpoint is a standard Triton health check,
			// usually not requiring authentication.
			url: '/v2/health/live',
			// If an API key is provided, the 'authenticate' block will attempt to add the Authorization header.
			// If the health endpoint itself is protected on a specific Triton setup,
			// this test will correctly fail if the key is missing or invalid.
		},
	};
}
