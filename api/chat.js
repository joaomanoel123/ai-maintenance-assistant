/**
 * ===============================================================================
 * CHAT API - Vercel Serverless Function
 * ===============================================================================
 * 
 * Endpoint de chat com HuggingFace Inference API
 * - Streaming de respostas
 * - Rate limiting
 * - Error handling
 * - CORS habilitado
 * 
 * Deploy: Vercel
 * Path: /api/chat
 * ===============================================================================
 */

import { HfInference } from "@huggingface/inference";

// Inicializar cliente HuggingFace
const client = new HfInference(process.env.HF_API_TOKEN);

// Configurações
const CONFIG = {
  model: process.env.HF_MODEL || "HuggingFaceH4/zephyr-7b-beta",
  maxTokens: parseInt(process.env.MAX_TOKENS) || 500,
  temperature: parseFloat(process.env.TEMPERATURE) || 0.7,
  topP: parseFloat(process.env.TOP_P) || 0.9,
};

// System prompt
const SYSTEM_PROMPT = `Você é uma AGI (Inteligência Artificial Geral) especializada em manutenção preditiva industrial.

CAPACIDADES:
- Predição de RUL (Remaining Useful Life)
- Predição de falhas industriais
- Análise de dados de sensores
- Recomendações técnicas

Seja técnico, direto e útil.`;

/**
 * Formatar prompt para o modelo
 */
function formatPrompt(message, history = []) {
  let prompt = `<|system|>\n${SYSTEM_PROMPT}</s>\n`;
  
  // Adicionar histórico (últimas 5 mensagens)
  const recentHistory = history.slice(-5);
  for (const msg of recentHistory) {
    if (msg.role === 'user') {
      prompt += `<|user|>\n${msg.content}</s>\n`;
    } else {
      prompt += `<|assistant|>\n${msg.content}</s>\n`;
    }
  }
  
  // Mensagem atual
  prompt += `<|user|>\n${message}</s>\n<|assistant|>`;
  
  return prompt;
}

/**
 * Validar requisição
 */
function validateRequest(body) {
  if (!body) {
    return { valid: false, error: "Body vazio" };
  }
  
  const { message } = body;
  
  if (!message || typeof message !== 'string') {
    return { valid: false, error: "Campo 'message' é obrigatório" };
  }
  
  if (message.length < 1 || message.length > 2000) {
    return { valid: false, error: "Mensagem deve ter entre 1 e 2000 caracteres" };
  }
  
  return { valid: true };
}

/**
 * Handler principal
 */
export default async function handler(req, res) {
  // CORS
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version'
  );
  
  // Preflight
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }
  
  // Apenas POST
  if (req.method !== 'POST') {
    return res.status(405).json({ 
      error: 'Method not allowed',
      allowedMethods: ['POST']
    });
  }
  
  try {
    // Parse body
    let body;
    try {
      body = typeof req.body === 'string' ? JSON.parse(req.body) : req.body;
    } catch (e) {
      return res.status(400).json({ error: 'JSON inválido' });
    }
    
    // Validar
    const validation = validateRequest(body);
    if (!validation.valid) {
      return res.status(400).json({ error: validation.error });
    }
    
    const { message, history = [], userId = 'anonymous', stream = false } = body;
    
    console.log(`📥 Mensagem de ${userId}: ${message.substring(0, 50)}...`);
    
    // Verificar token
    if (!process.env.HF_API_TOKEN) {
      console.error('❌ HF_API_TOKEN não configurado');
      return res.status(500).json({ 
        error: 'Configuração inválida',
        message: 'API token não configurado'
      });
    }
    
    // Formatar prompt
    const prompt = formatPrompt(message, history);
    
    // Streaming ou resposta completa
    if (stream) {
      // Streaming (SSE)
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      try {
        const stream = client.textGenerationStream({
          model: CONFIG.model,
          inputs: prompt,
          parameters: {
            max_new_tokens: CONFIG.maxTokens,
            temperature: CONFIG.temperature,
            top_p: CONFIG.topP,
            return_full_text: false,
          }
        });
        
        for await (const chunk of stream) {
          if (chunk.token?.text) {
            res.write(`data: ${JSON.stringify({ token: chunk.token.text })}\n\n`);
          }
        }
        
        res.write('data: [DONE]\n\n');
        res.end();
        
      } catch (streamError) {
        console.error('Erro no streaming:', streamError);
        res.write(`data: ${JSON.stringify({ error: streamError.message })}\n\n`);
        res.end();
      }
      
    } else {
      // Resposta completa
      const result = await client.textGeneration({
        model: CONFIG.model,
        inputs: prompt,
        parameters: {
          max_new_tokens: CONFIG.maxTokens,
          temperature: CONFIG.temperature,
          top_p: CONFIG.topP,
          return_full_text: false,
        }
      });
      
      const reply = result.generated_text.trim();
      
      console.log(`✅ Resposta gerada: ${reply.substring(0, 50)}...`);
      
      return res.status(200).json({ 
        reply,
        model: CONFIG.model,
        timestamp: new Date().toISOString()
      });
    }
    
  } catch (error) {
    console.error('❌ Erro na API:', error);
    
    // Tratamento de erros específicos
    if (error.message?.includes('rate limit')) {
      return res.status(429).json({ 
        error: 'Rate limit excedido',
        message: 'Muitas requisições. Tente novamente em alguns segundos.',
        retryAfter: 60
      });
    }
    
    if (error.message?.includes('model')) {
      return res.status(503).json({ 
        error: 'Modelo indisponível',
        message: 'O modelo está sendo carregado. Tente novamente em 20 segundos.',
        retryAfter: 20
      });
    }
    
    return res.status(500).json({ 
      error: 'Erro interno',
      message: error.message || 'Erro desconhecido'
    });
  }
}

/**
 * Configuração do Vercel
 */
export const config = {
  api: {
    bodyParser: {
      sizeLimit: '1mb',
    },
    externalResolver: true,
  },
  maxDuration: 60, // 60 segundos (Pro plan)
};
