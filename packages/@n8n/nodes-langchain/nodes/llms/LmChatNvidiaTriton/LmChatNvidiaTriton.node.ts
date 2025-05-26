/* eslint-disable n8n-nodes-base/node-dirname-against-convention */
import {
	NodeConnectionTypes,
	type INodeType,
	type INodeTypeDescription,
	type ISupplyDataFunctions,
	type ILoadOptionsFunctions,
	type INodeProperties,
	type SupplyData,
} from 'n8n-workflow';
import { getConnectionHintNoticeField } from '@utils/sharedFields';

export class LmChatNvidiaTriton implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'NVIDIA Triton Chat Model',
		name: 'lmChatNvidiaTriton',
		icon: 'file:nvidia-7.svg',
		group: ['transform'],
		version: 1,
		description: 'Send prompts to an NVIDIA Triton Inference Server Generate Extension',
		defaults: {
			name: 'NVIDIA Triton Chat',
		},
		inputs: [],
		outputs: [NodeConnectionTypes.AiLanguageModel],
		outputNames: ['Model'],
		credentials: [
			{
				name: 'nvidiaTritonApi',
				required: true,
			},
		],
		properties: [
			// hint when attached to an AI chain or agent
			getConnectionHintNoticeField([
				NodeConnectionTypes.AiChain,
				NodeConnectionTypes.AiAgent,
			]),
			{
				displayName: 'Model Name',
				name: 'modelName',
				type: 'options',
				typeOptions: {
					loadOptionsMethod: 'getTritonModels',
				},
				default: '',
				description:
					'Select the model deployed on Triton. It must support the `/generate` extension.',
			},
			{
				displayName: 'Options',
				name: 'options',
				type: 'collection',
				placeholder: 'Add Option',
				default: {},
				options: [
					{
						displayName: 'Max New Tokens',
						name: 'maxTokensToSample',
						type: 'number',
						default: 512,
						description:
							'Maximum number of tokens to generate (if supported by your model).',
						typeOptions: { minValue: 1 },
					},
					{
						displayName: 'Temperature',
						name: 'temperature',
						type: 'number',
						default: 0.7,
						description:
							'Sampling temperature (0–1+). Lower is more deterministic.',
						typeOptions: {
							minValue: 0,
							maxValue: 2,
							numberPrecision: 2,
						},
					},
					{
						displayName: 'Stream Response',
						name: 'stream',
						type: 'boolean',
						default: false,
						description:
							'Whether to request streamed output (model/server must support it).',
					},
				],
			},
		],
	};

	methods = {
		loadOptions: {
			/**
			 * Fetches models from Triton’s repository index and returns only READY ones.
			 */
			async getTritonModels(
				this: ILoadOptionsFunctions,
			): Promise<{ name: string; value: string }[]> {
				const creds = await this.getCredentials('nvidiaTritonApi');
				const serverUrl = (creds.serverUrl as string).replace(/\/$/, '');
				if (!serverUrl) {
					throw new Error('Please configure the Triton Server URL in credentials.');
				}

				const endpoint = `${serverUrl}/v2/repository/index`;
				try {
					const data = await this.helpers.httpRequest({
						method: 'GET',
						url: endpoint,
						json: true,
					});
					if (Array.isArray(data)) {
						const opts = data
							.filter((m: any) => m.state === 'READY' && m.name)
							.map((m: any) => ({ name: m.name, value: m.name }));
						if (opts.length) return opts;
					}
					return [{ name: 'No READY models found', value: '' }];
				} catch (err: any) {
					return [{ name: `Error: ${err.message}`, value: '' }];
				}
			},
		},
	};

	/**
	 * supplyData returns an object under `response` that downstream AI‐chain nodes
	 * will call via `generate(messages)` to get back LLMResult‐shaped output.
	 */
	async supplyData(
		this: ISupplyDataFunctions,
		itemIndex: number,
	): Promise<SupplyData> {
		const creds = await this.getCredentials('nvidiaTritonApi');
		const serverUrl = (creds.serverUrl as string).replace(/\/$/, '');
		const apiKey = creds.apiKey as string; // optional

		if (!serverUrl) {
			throw new Error('Triton Server URL is not defined in credentials.');
		}

		const modelName = this.getNodeParameter('modelName', itemIndex) as string;
		const opts = this.getNodeParameter('options', itemIndex, {}) as {
			maxTokensToSample?: number;
			temperature?: number;
			stream?: boolean;
		};

		// Minimal model interface expected by n8n AI‐chain consumer
		const model = {
			/**
			 * messages: array of { role, content }
			 * returns: { generations: [[{ text }]] }
			 */
			generate: async (messages: { role: string; content: string }[]) => {
				// build prompt
				const prompt = messages
					.map((m) => `${m.role}: ${m.content}`)
					.join('\n');
				// Triton expects { text_input, parameters: {...} }
				const body: any = {
					text_input: prompt,
					parameters: {
						stream: opts.stream ?? false,
						temperature: opts.temperature ?? 0,
					},
				};
				if (opts.maxTokensToSample) {
					body.parameters.max_new_tokens = opts.maxTokensToSample;
				}

				// headers
				const headers: Record<string, string> = {
					'Content-Type': 'application/json',
				};
				if (apiKey) {
					headers.Authorization = `Bearer ${apiKey}`;
				}

				// call Triton
				const response = await this.helpers.httpRequest({
					method: 'POST',
					url: `${serverUrl}/v2/models/${modelName}/generate`,
					headers,
					body,
					json: true,
				});

				// extract text
				let text = '';
				if (typeof response.generated_text === 'string') {
					text = response.generated_text;
				} else if (
					Array.isArray(response.data) &&
					typeof response.data[0] === 'string'
				) {
					text = response.data[0];
				} else if (typeof response.data === 'string') {
					text = response.data;
				} else {
					throw new Error(
						`Unexpected Triton response: ${JSON.stringify(response)}`,
					);
				}

				return {
					generations: [[{ text }]],
				};
			},
		};

		return {
			response: model,
		};
	}
}
