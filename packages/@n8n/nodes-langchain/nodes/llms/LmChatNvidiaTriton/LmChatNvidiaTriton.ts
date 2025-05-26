// LmNvidiaTritonChat.node.ts

import type {
	IExecuteFunctions,
	INodeExecutionData,
	INodeType,
	INodeTypeDescription,
	INodeProperties,
	ILoadOptionsFunctions,
	INodePropertyOptions,
} from 'n8n-workflow';

// Interface for NVIDIA Triton API Credentials
// This will be defined in a separate credentials file (e.g., NvidiaTritonApi.credentials.ts)
export interface INvidiaTritonApiCredentials {
	serverUrl: string;
	// apiKey?: string; // Optional: if your Triton server uses API key authentication
}

export class LmNvidiaTritonChat implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'NVIDIA Triton Generate', // Updated display name
		name: 'lmNvidiaTritonGenerate', // Updated name to reflect /generate endpoint
		icon: 'file:nvidiaTriton.svg', // You will need to create this SVG file
		group: ['ai'],
		version: 1,
		subtitle: '={{$parameter["modelName"]}}',
		description: 'Sends requests to an NVIDIA Triton Server model via the /generate endpoint',
		defaults: {
			name: 'NVIDIA Triton Generate',
		},
		credentials: [
			{
				name: 'nvidiaTritonApi',
				required: true,
				description: 'NVIDIA Triton Server connection',
			},
		],
		inputs: ['main'],
		outputs: ['main'],
		properties: [
			// --- Core Parameters ---
			{
				displayName: 'Model Name',
				name: 'modelName',
				type: 'options',
				default: '',
				required: true,
				description:
					'Name of the model deployed on the Triton server (e.g., "llama2vllm"). <a href="https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md" target="_blank">Triton Model Repository API</a>',
				typeOptions: {
					loadOptionsMethod: 'getModels',
				},
			},
			// Model Version is less commonly used in the /generate endpoint path,
			// but kept for completeness if needed in parameters.
			{
				displayName: 'Model Version (Optional)',
				name: 'modelVersion',
				type: 'string',
				default: '',
				description: 'Version of the model. Note: Typically not used in the /generate endpoint URL path, but can be passed as a parameter if the model supports it.',
				placeholder: '1',
			},
			{
				displayName: 'Text Input',
				name: 'textInput',
				type: 'string',
				default: '',
				required: true,
				typeOptions: {
					rows: 4,
				},
				description: 'The primary input string for the model.',
				placeholder: 'What is Triton Inference Server?',
			},

			// --- Additional Inference Parameters ---
			{
				displayName: 'Parameters',
				name: 'parameters', // Changed from additionalParameters for clarity
				type: 'collection',
				placeholder: 'Add Parameter',
				default: {
					// Default common parameters based on the curl example
					options: [
						{ name: 'stream', value: false },
						{ name: 'temperature', value: 0.7 },
						{ name: 'max_tokens', value: 256 }, // Common name, might be 'max_new_tokens' for some models
					]
				},
				description: 'Parameters to include in the request (e.g., stream, temperature, max_tokens). These will be sent in the nested "parameters" object.',
				options: [
					{
						displayName: 'Parameter Name',
						name: 'name',
						type: 'string',
						default: '',
						description: 'Name of the parameter (e.g., stream, temperature, top_p, max_tokens).',
					},
					{
						displayName: 'Parameter Value',
						name: 'value',
						type: 'string', // Keep as string for flexibility, user can input numbers/booleans
						default: '',
						description: 'Value for the parameter. Enter true/false for booleans, numbers for numeric values.',
					},
				],
			},

			// --- Advanced Configuration ---
			{
				displayName: 'Advanced Options',
				name: 'advancedOptions',
				type: 'boolean',
				default: false,
				description: 'Show advanced parameters for request field names and endpoint path.',
			},
			{
				displayName: 'API Endpoint Path Segment',
				name: 'endpointPathSegment',
				type: 'string',
				default: '/generate',
				required: true,
				displayOptions: {
					show: {
						advancedOptions: [true],
					},
				},
				description: 'The API path segment for the model action (e.g., /generate, /infer).',
				placeholder: '/generate',
			},
			{
				displayName: 'Request Input Field Name',
				name: 'requestInputField',
				type: 'string',
				default: 'text_input',
				required: true,
				displayOptions: {
					show: {
						advancedOptions: [true],
					},
				},
				description: 'Name of the field in the request payload that holds the main text input.',
			},
			{
				displayName: 'Response Output Field Name',
				name: 'responseOutputField',
				type: 'string',
				default: 'text_output',
				required: true,
				displayOptions: {
					show: {
						advancedOptions: [true],
					},
				},
				description: 'Name of the field in the response payload that contains the generated text.',
			},

			// --- Output Configuration ---
			{
				displayName: 'Output Field Name (n8n Item)',
				name: 'outputFieldNameN8n',
				type: 'string',
				default: 'completion',
				description: 'Name of the field in the n8n output item to store the model\'s primary response text.',
			},
		],
	};

	methods = {
		loadOptions: {
			async getModels(this: ILoadOptionsFunctions): Promise<INodePropertyOptions[]> {
				const credentials = (await this.getCredentials(
					'nvidiaTritonApi',
				)) as INvidiaTritonApiCredentials | undefined;

				if (!credentials?.serverUrl) {
					return [{ name: 'Error: Triton Server URL not configured in credentials.', value: '' }];
				}

				const serverUrl = credentials.serverUrl.replace(/\/+$/, '');
				const endpoint = `${serverUrl}/v2/repository/index`; // This endpoint is for model discovery

				try {
					const response = await this.helpers.httpRequest({
						method: 'GET',
						url: endpoint,
						json: true,
					});

					if (Array.isArray(response)) {
						const models: INodePropertyOptions[] = response
							.filter((model: any) => model.name && (model.state === 'READY' || !model.state)) // Filter for ready models
							.map((model: any) => ({
								name: model.version ? `${model.name} (v${model.version})` : model.name,
								value: model.name,
							}));
						const uniqueModels = models.filter(
							(model, index, self) => index === self.findIndex((m) => m.value === model.value),
						);
						if (uniqueModels.length === 0) return [{ name: 'No ready models found or repository index empty.', value: '' }];
						return uniqueModels;
					}
					console.warn('Unexpected response format from Triton model repository:', response);
					return [{ name: 'Error fetching models (unexpected format)', value: '' }];
				} catch (error) {
					let errorMessage = 'Error fetching models. Check Triton server URL and network.';
					if (error.message?.includes('ECONNREFUSED')) errorMessage = `Connection refused at ${serverUrl}.`;
					else if (error.response?.data) errorMessage = `Error from Triton: ${JSON.stringify(error.response.data) || error.message}`;
					else if (error.message) errorMessage = `Network/other error: ${error.message}`;
					console.error('Error fetching Triton models:', error);
					return [{ name: errorMessage, value: '' }];
				}
			},
		},
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const returnData: INodeExecutionData[] = [];

		const credentials = (await this.getCredentials(
			'nvidiaTritonApi',
		)) as INvidiaTritonApiCredentials;
		const serverUrl = credentials.serverUrl.replace(/\/+$/, '');

		for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
			try {
				const modelName = this.getNodeParameter('modelName', itemIndex, '') as string;
				// const modelVersion = this.getNodeParameter('modelVersion', itemIndex, '') as string | undefined; // Kept if needed for parameters object
				const textInputValue = this.getNodeParameter('textInput', itemIndex, '') as string;
				const outputFieldNameN8n = this.getNodeParameter('outputFieldNameN8n', itemIndex, 'completion') as string;

				const advancedOptions = this.getNodeParameter('advancedOptions', itemIndex, false) as boolean;
				const endpointPathSegment = this.getNodeParameter('endpointPathSegment', itemIndex, '/generate') as string;
				const requestInputField = this.getNodeParameter('requestInputField', itemIndex, 'text_input') as string;
				const responseOutputField = this.getNodeParameter('responseOutputField', itemIndex, 'text_output') as string;

				const paramsCollection = this.getNodeParameter('parameters.options', itemIndex, []) as Array<{name: string, value: any}>;
				const tritonApiParameters: Record<string, any> = {};
				for (const param of paramsCollection) {
					if (param.name) {
						// Attempt to parse boolean/numeric strings
						let value = param.value;
						if (typeof param.value === 'string') {
							if (param.value.toLowerCase() === 'true') value = true;
							else if (param.value.toLowerCase() === 'false') value = false;
							else if (!isNaN(parseFloat(param.value)) && isFinite(param.value as any)) value = parseFloat(param.value);
						}
						tritonApiParameters[param.name] = value;
					}
				}

				if (!modelName) throw new Error('Model Name is a required parameter.');
				if (!textInputValue && requestInputField) throw new Error('Text Input is a required parameter.'); // Check if input field is expected

				// Construct URL: server/v2/models/MODEL_NAME/generate (or other segment)
				// Model version is typically not part of the /generate path.
				const apiUrl = `${serverUrl}/v2/models/${encodeURIComponent(modelName)}${endpointPathSegment.startsWith('/') ? endpointPathSegment : '/' + endpointPathSegment}`;

				const payload: Record<string, any> = {};
				if (requestInputField) {
					payload[requestInputField] = textInputValue;
				}
				if (Object.keys(tritonApiParameters).length > 0) {
					payload.parameters = tritonApiParameters;
				}
				// If modelVersion is provided and needs to be in parameters:
				// const modelVersionParam = this.getNodeParameter('modelVersion', itemIndex, '') as string;
				// if (modelVersionParam && payload.parameters) {
				//    payload.parameters.model_version = modelVersionParam; // Or however model expects it
				// } else if (modelVersionParam) {
				//    payload.parameters = { model_version: modelVersionParam };
				// }


				const responseData = await this.helpers.httpRequest({
					method: 'POST',
					url: apiUrl,
					body: payload,
					json: true,
					// headers: { 'Authorization': `Bearer ${credentials.apiKey}` }, // If API key needed
				});

				let completionText = '';
				let rawResponse: any = responseData;

				if (responseData && typeof responseData === 'object' && responseData !== null && responseOutputField in responseData) {
					completionText = String(responseData[responseOutputField]);
				} else {
					const message = `Expected output field '${responseOutputField}' not found in Triton response or response is not an object.`;
					console.warn(message, responseData);
					// completionText = `Error: ${message}`; // Avoid setting error directly into completion
					// Instead, the raw response will be in the output for debugging.
					if (this.continueOnFail()) {
						// If continue on fail, we might want to pass the raw response or a structured error.
						// For now, completionText remains empty or indicates an issue if not found.
					} else {
						// If not continuing on fail, an error should be thrown if the output is critical.
						// However, the try/catch will handle this.
						// For now, we just log and the raw response is passed.
					}
				}

				const resultItem: INodeExecutionData = {
					json: {
						...items[itemIndex].json,
						[outputFieldNameN8n]: completionText,
						rawTritonResponse: rawResponse,
					},
					pairedItem: { item: itemIndex },
				};
				returnData.push(resultItem);

			} catch (error) {
				const errorDetails = {
					message: error.message,
					stack: error.stack,
					details: error.response?.data || error.cause || 'No additional details',
					requestUrl: error.request?.url, // If available from httpRequest error
					requestPayload: error.request?.body, // If available
				};
				if (this.continueOnFail()) {
					const errorData = {
						json: items[itemIndex].json,
						error: errorDetails,
						pairedItem: { item: itemIndex },
					};
					returnData.push(errorData as INodeExecutionData);
					continue;
				}
				// Re-throw with more context if possible, or let n8n handle the original error
				error.message = `Triton API Error: ${error.message}. Check node parameters and Triton server logs.`;
				if (error.response?.data?.error) { // Triton often has a nested error message
					error.message += ` Server detail: ${error.response.data.error}`;
				}
				throw error;
			}
		}
		return this.prepareOutputData(returnData);
	}
}

