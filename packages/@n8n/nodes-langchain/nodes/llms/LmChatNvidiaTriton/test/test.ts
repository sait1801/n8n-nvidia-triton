// LmNvidiaTritonChat.node.test.ts

import {
	type IExecuteFunctions,
	type ILoadOptionsFunctions,
	type INodeExecutionData,
	type INodePropertyOptions,
	NodeApiError,
} from 'n8n-workflow';

// Import the class from your node file
import { LmNvidiaTritonChat, type INvidiaTritonApiCredentials } from './LmNvidiaTritonChat.node.ts'; // Adjust path if necessary

describe('LmNvidiaTritonChat Node', () => {
	let nodeInstance: LmNvidiaTritonChat;

	// Mock IExecuteFunctions
	const mockExecuteFunctions: IExecuteFunctions = {
		// @ts-ignore - We only mock what's needed for the test
		getCredentials: jest.fn(),
		// @ts-ignore
		getNodeParameter: jest.fn(),
		getInputData: jest.fn(),
		// @ts-ignore
		helpers: {
			httpRequest: jest.fn(),
			// @ts-ignore
			constructExecutionMetaData: jest.fn((data, meta) => data.map(item => ({ ...item, ...meta?.itemData }))), // Simplified mock
		},
		// @ts-ignore
		prepareOutputData: jest.fn((data) => data), // Pass through for testing
		// @ts-ignore
		continueOnFail: jest.fn().mockReturnValue(false),
		// Add other methods if your node uses them directly
	};

	// Mock ILoadOptionsFunctions
	const mockLoadOptionsFunctions: ILoadOptionsFunctions = {
		// @ts-ignore
		getCredentials: jest.fn(),
		// @ts-ignore
		getCurrentNodeParameter: jest.fn(),
		// @ts-ignore
		helpers: {
			httpRequest: jest.fn(),
		},
		// Add other methods if your node uses them directly
	};


	beforeEach(() => {
		// Reset mocks before each test
		jest.clearAllMocks();

		// Create a new instance of the node for each test
		nodeInstance = new LmNvidiaTritonChat();

		// Default mock for getCredentials
		(mockExecuteFunctions.getCredentials as jest.Mock).mockResolvedValue({
			serverUrl: 'http://mock-triton:8000',
		} as INvidiaTritonApiCredentials);
		(mockLoadOptionsFunctions.getCredentials as jest.Mock).mockResolvedValue({
			serverUrl: 'http://mock-triton:8000',
		} as INvidiaTritonApiCredentials);

		// Default mock for getInputData
		(mockExecuteFunctions.getInputData as jest.Mock).mockReturnValue([{ json: { initialData: 'test' } }] as INodeExecutionData[]);
	});

	describe('execute', () => {
		it('should successfully call Triton /generate endpoint and return data', async () => {
			// Setup parameters for this test case
			(mockExecuteFunctions.getNodeParameter as jest.Mock)
				.mockImplementation((name: string, itemIndex: number, defaultValue?: any) => {
					if (name === 'modelName') return 'llama2vllm';
					if (name === 'textInput') return 'What is Triton?';
					if (name === 'parameters') return { options: [{ name: 'stream', value: 'false' }, { name: 'temperature', value: '0.1' }] };
					if (name === 'outputFieldNameN8n') return 'completion';
					if (name === 'advancedOptions') return false;
					if (name === 'endpointPathSegment') return '/generate';
					if (name === 'requestInputField') return 'text_input';
					if (name === 'responseOutputField') return 'text_output';
					return defaultValue;
				});

			const mockTritonResponse = {
				model_name: 'llama2vllm',
				model_version: '1',
				text_output: 'Triton is an inference server.',
			};
			(mockExecuteFunctions.helpers.httpRequest as jest.Mock).mockResolvedValue(mockTritonResponse);

			const result = await nodeInstance.execute.call(mockExecuteFunctions);

			expect(mockExecuteFunctions.helpers.httpRequest).toHaveBeenCalledWith({
				method: 'POST',
				url: 'http://mock-triton:8000/v2/models/llama2vllm/generate',
				body: {
					text_input: 'What is Triton?',
					parameters: {
						stream: false,
						temperature: 0.1,
					},
				},
				json: true,
			});
			expect(result).toHaveLength(1);
			expect(result[0][0].json.initialData).toBe('test'); // Check if input data is preserved
			expect(result[0][0].json.completion).toBe('Triton is an inference server.');
			expect(result[0][0].json.rawTritonResponse).toEqual(mockTritonResponse);
		});

		it('should handle API errors from Triton when continueOnFail is false', async () => {
			(mockExecuteFunctions.getNodeParameter as jest.Mock).mockReturnValueOnce('test-model').mockReturnValueOnce('test-prompt'); // modelName, textInput
			(mockExecuteFunctions.helpers.httpRequest as jest.Mock).mockRejectedValue(
				new NodeApiError(mockExecuteFunctions as any, { message: 'Triton server error', httpCode: '500' } as any, { response: { data: { error: 'Internal server error' } } }),
			);
			(mockExecuteFunctions.continueOnFail as jest.Mock).mockReturnValue(false);

			await expect(nodeInstance.execute.call(mockExecuteFunctions)).rejects.toThrow(/Triton API Error: Triton server error. Server detail: Internal server error/);
		});

		it('should return error data when continueOnFail is true', async () => {
			(mockExecuteFunctions.getNodeParameter as jest.Mock).mockReturnValueOnce('test-model').mockReturnValueOnce('test-prompt');
			const error = new NodeApiError(mockExecuteFunctions as any, { message: 'Triton server error', httpCode: '500' } as any, { response: { data: { error: 'Internal server error' } } });
			(mockExecuteFunctions.helpers.httpRequest as jest.Mock).mockRejectedValue(error);
			(mockExecuteFunctions.continueOnFail as jest.Mock).mockReturnValue(true);

			const result = await nodeInstance.execute.call(mockExecuteFunctions);

			expect(result).toHaveLength(1);
			expect(result[0][0].json.initialData).toBe('test');
			expect(result[0][0].error).toBeDefined();
			expect(result[0][0].error?.message).toContain('Triton server error');
		});

		it('should use advanced options for field names and endpoint', async () => {
			(mockExecuteFunctions.getNodeParameter as jest.Mock)
				.mockImplementation((name: string, itemIndex: number, defaultValue?: any) => {
					if (name === 'modelName') return 'customModel';
					if (name === 'textInput') return 'Custom input';
					if (name === 'parameters') return { options: [] };
					if (name === 'outputFieldNameN8n') return 'customN8nOutput';
					if (name === 'advancedOptions') return true;
					if (name === 'endpointPathSegment') return '/custom_generate';
					if (name === 'requestInputField') return 'my_text_in';
					if (name === 'responseOutputField') return 'my_text_out';
					return defaultValue;
				});

			const mockTritonResponse = { my_text_out: 'Custom output text.' };
			(mockExecuteFunctions.helpers.httpRequest as jest.Mock).mockResolvedValue(mockTritonResponse);

			const result = await nodeInstance.execute.call(mockExecuteFunctions);

			expect(mockExecuteFunctions.helpers.httpRequest).toHaveBeenCalledWith(
				expect.objectContaining({
					url: 'http://mock-triton:8000/v2/models/customModel/custom_generate',
					body: {
						my_text_in: 'Custom input',
						parameters: {},
					},
				}),
			);
			expect(result[0][0].json.customN8nOutput).toBe('Custom output text.');
		});

		it('should throw error if modelName is missing', async () => {
			(mockExecuteFunctions.getNodeParameter as jest.Mock)
				.mockImplementation((name: string) => {
					if (name === 'modelName') return ''; // Missing modelName
					if (name === 'textInput') return 'Some input';
					return undefined;
				});

			await expect(nodeInstance.execute.call(mockExecuteFunctions)).rejects.toThrow('Model Name is a required parameter.');
		});
	});

	describe('getModels (loadOptions)', () => {
		it('should fetch and return a list of models', async () => {
			const mockRepoIndex = [
				{ name: 'model1', version: '1', state: 'READY' },
				{ name: 'model2', state: 'READY' },
				{ name: 'model1', version: '2', state: 'LOADING' }, // Should be filtered out if not READY
				{ name: 'model3' }, // No state, should be included
			];
			(mockLoadOptionsFunctions.helpers.httpRequest as jest.Mock).mockResolvedValue(mockRepoIndex);

			const result = await nodeInstance.methods!.loadOptions!.getModels.call(mockLoadOptionsFunctions);

			expect(mockLoadOptionsFunctions.helpers.httpRequest).toHaveBeenCalledWith({
				method: 'GET',
				url: 'http://mock-triton:8000/v2/repository/index',
				json: true,
			});
			expect(result).toEqual([
				{ name: 'model1 (v1)', value: 'model1' },
				{ name: 'model2', value: 'model2' },
				{ name: 'model3', value: 'model3' },
			]);
		});

		it('should handle errors when fetching models', async () => {
			(mockLoadOptionsFunctions.helpers.httpRequest as jest.Mock).mockRejectedValue(new Error('Network error'));

			const result = await nodeInstance.methods!.loadOptions!.getModels.call(mockLoadOptionsFunctions);

			expect(result).toEqual([{ name: 'Network or other error: Network error', value: '' }]);
		});

		it('should handle empty model list from server', async () => {
			(mockLoadOptionsFunctions.helpers.httpRequest as jest.Mock).mockResolvedValue([]);

			const result = await nodeInstance.methods!.loadOptions!.getModels.call(mockLoadOptionsFunctions);

			expect(result).toEqual([{ name: 'No ready models found or repository index empty.', value: '' }]);
		});

		it('should return error if credentials are not configured', async () => {
			(mockLoadOptionsFunctions.getCredentials as jest.Mock).mockResolvedValue(undefined); // No credentials

			const result = await nodeInstance.methods!.loadOptions!.getModels.call(mockLoadOptionsFunctions);
			expect(result).toEqual([{ name: 'Error: Triton Server URL not configured in credentials.', value: '' }]);
		});
	});
});

