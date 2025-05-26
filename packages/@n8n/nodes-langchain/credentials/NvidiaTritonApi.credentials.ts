import type {
	// IAuthenticateGeneric, // Not used as Triton /generate often doesn't require auth headers by default
	ICredentialTestRequest,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class NvidiaTritonApi implements ICredentialType {
	name = 'nvidiaTritonApi'; // This name is used to reference the credential in nodes

	displayName = 'Nvidia Triton API';

	// Link to official Triton documentation
	documentationUrl = 'https://developer.nvidia.com/triton-inference-server';

	properties: INodeProperties[] = [
		{
			displayName: 'Triton Server Base URL',
			name: 'baseUrl',
			type: 'string',
			required: true,
			default: 'http://localhost:8000', // Common default for local Triton
			description:
				'The base URL of your Nvidia Triton Inference Server (e.g., http://localhost:8000). This should be the URL up to, but not including, /v2/... paths.',
			placeholder: 'http://<your-triton-server-address>:<port>',
		},
		// If your Triton server requires an API key or other auth, you could add properties here:
		// {
		// 	displayName: 'API Key',
		// 	name: 'apiKey',
		// 	type: 'string',
		// 	typeOptions: { password: true },
		// 	default: '',
		// 	description: 'Optional API Key if your Triton server is secured with one.',
		// },
	];

	// The /generate endpoint example provided didn't use bearer token authentication.
	// If your Triton setup requires a specific authentication header (like an API key),
	// you would define the 'authenticate' property. For example:
	// authenticate: IAuthenticateGeneric = {
	// 	type: 'generic',
	// 	properties: {
	// 		headers: {
	// 			'Authorization': '=Bearer {{$credentials.apiKey}}', // Or 'X-Api-Key': '={{$credentials.apiKey}}'
	// 		},
	// 	},
	// };
	// For now, since the cURL example was unauthenticated, we omit it.
	// The node itself can add headers if needed, or this credential can be expanded.

	test: ICredentialTestRequest = {
		request: {
			// The baseURL will be dynamically set from the 'baseUrl' property defined above.
			// n8n will interpolate {{$credentials.baseUrl}} automatically.
			baseURL: '={{$credentials.baseUrl}}',
			// Using a standard Triton health check endpoint.
			// This request will be sent to: {{$credentials.baseUrl}}/v2/health/live
			url: '/v2/health/live',
			method: 'GET', // Standard method for health checks
			// If your health endpoint required specific headers (even for an unauthenticated one),
			// they could be added here. For /v2/health/live, usually none are needed.
			// headers: {},
		},
	};
}
