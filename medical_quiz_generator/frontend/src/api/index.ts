import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1'

export const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
})

// Types
export interface Document {
    id: string
    filename: string
    title: string
    description?: string
    specialty?: string
    tags: string[]
    file_type: string
    file_size: number
    num_pages?: number
    num_chunks: number
    status: 'pending' | 'processing' | 'completed' | 'failed'
    created_at: string
    error?: string
}

export interface QuestionOption {
    id: string
    text: string
    is_correct: boolean
}

export interface AIReview {
    status: 'approved' | 'needs_revision' | 'reject' | 'skipped' | 'error'
    accuracy_score?: number
    clarity_score?: number
    educational_value?: number
    issues: string[]
    suggestions: string[]
    corrected_answer?: string
    corrected_explanation?: string
    reviewed: boolean
    error?: string
}

export interface Question {
    id: string
    question_text: string
    question_type: 'single_choice' | 'multiple_choice' | 'true_false' | 'case_based'
    options: QuestionOption[]
    correct_answer: string
    explanation: string
    difficulty: 'easy' | 'medium' | 'hard'
    topic?: string
    keywords?: string[]
    document_id?: string
    reference_text?: string
    created_at: string
    ai_review?: AIReview
}

export interface GenerationRequest {
    document_ids: string[]
    num_questions: number
    difficulty?: 'easy' | 'medium' | 'hard'
    question_types?: string[]
    topics?: string[]
    focus_areas?: string[]
    include_case_based?: boolean
    include_explanations?: boolean
    enable_double_check?: boolean
}

export interface ReviewStats {
    total_questions: number
    reviewed: number
    high_accuracy: number
    needs_revision: number
    review_rate: number
}

export interface GenerationStatus {
    task_id: string
    status: 'pending' | 'processing' | 'completed' | 'failed'
    progress: number
    total_questions: number
    generated_questions: number
    questions?: Question[]
    review_stats?: ReviewStats
    error?: string
}

export interface SearchResult {
    chunk_id: string
    document_id: string
    content: string
    score: number
    metadata: Record<string, unknown>
}

// Document APIs
export const documentsApi = {
    upload: async (file: File, metadata: { title?: string; description?: string; specialty?: string; tags?: string }) => {
        const formData = new FormData()
        formData.append('file', file)
        if (metadata.title) formData.append('title', metadata.title)
        if (metadata.description) formData.append('description', metadata.description)
        if (metadata.specialty) formData.append('specialty', metadata.specialty)
        if (metadata.tags) formData.append('tags', metadata.tags)

        const response = await api.post('/documents/upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        })
        return response.data
    },

    list: async (params?: { specialty?: string; status?: string; limit?: number; offset?: number }) => {
        const response = await api.get('/documents/', { params })
        return response.data
    },

    get: async (id: string) => {
        const response = await api.get(`/documents/${id}`)
        return response.data
    },

    delete: async (id: string) => {
        const response = await api.delete(`/documents/${id}`)
        return response.data
    },

    getChunks: async (id: string) => {
        const response = await api.get(`/documents/${id}/chunks`)
        return response.data
    },

    getStats: async () => {
        const response = await api.get('/documents/stats/overview')
        return response.data
    },
}

// Question APIs
export const questionsApi = {
    generate: async (request: GenerationRequest) => {
        const response = await api.post('/questions/generate', request)
        return response.data
    },

    getGenerationStatus: async (taskId: string) => {
        const response = await api.get(`/questions/generate/${taskId}/status`)
        return response.data as GenerationStatus
    },

    list: async (params?: {
        document_id?: string
        difficulty?: string
        question_type?: string
        topic?: string
        limit?: number
        offset?: number
    }) => {
        const response = await api.get('/questions/', { params })
        return response.data
    },

    get: async (id: string) => {
        const response = await api.get(`/questions/${id}`)
        return response.data
    },

    update: async (id: string, updates: Partial<Question>) => {
        const response = await api.put(`/questions/${id}`, updates)
        return response.data
    },

    delete: async (id: string) => {
        const response = await api.delete(`/questions/${id}`)
        return response.data
    },

    search: async (query: string, documentIds?: string[], topK?: number) => {
        const response = await api.post('/questions/search', {
            query,
            document_ids: documentIds,
            top_k: topK || 5,
        })
        return response.data
    },

    export: async (questionIds: string[], format: 'json' | 'pdf' | 'docx' | 'excel', options?: {
        include_answers?: boolean
        include_explanations?: boolean
        shuffle_questions?: boolean
        shuffle_options?: boolean
    }) => {
        const response = await api.post('/questions/export', {
            question_ids: questionIds,
            format,
            ...options,
        })
        return response.data
    },

    getStats: async () => {
        const response = await api.get('/questions/stats/overview')
        return response.data
    },
}

// Config APIs
export const configApi = {
    getSpecialties: async () => {
        const response = await api.get('/specialties')
        return response.data.specialties as string[]
    },

    getConfig: async () => {
        const response = await api.get('/config')
        return response.data
    },
}
