import { useEffect, useState, useCallback, useMemo, useRef } from 'react'
import { SigmaContainer, useRegisterEvents, useSigma } from '@react-sigma/core'
import { Settings as SigmaSettings } from 'sigma/settings'
import { GraphSearchOption, OptionItem } from '@react-sigma/graph-search'
import { EdgeArrowProgram, NodePointProgram, NodeCircleProgram } from 'sigma/rendering'
import { NodeBorderProgram } from '@sigma/node-border'
import { EdgeCurvedArrowProgram, createEdgeCurveProgram } from '@sigma/edge-curve'

// UI 組件
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/Select"
import { Database } from "lucide-react"

import FocusOnNode from '@/components/graph/FocusOnNode'
import LayoutsControl from '@/components/graph/LayoutsControl'
import GraphControl from '@/components/graph/GraphControl'
import ZoomControl from '@/components/graph/ZoomControl'
import FullScreenControl from '@/components/graph/FullScreenControl'
import Settings from '@/components/graph/Settings'
import GraphSearch from '@/components/graph/GraphSearch'
import GraphLabels from '@/components/graph/GraphLabels'
import PropertiesView from '@/components/graph/PropertiesView'
import SettingsDisplay from '@/components/graph/SettingsDisplay'
import Legend from '@/components/graph/Legend'
import LegendButton from '@/components/graph/LegendButton'

import { useSettingsStore } from '@/stores/settings'
import { useGraphStore } from '@/stores/graph'
import { labelColorDarkTheme, labelColorLightTheme } from '@/lib/constants'
import { getWorkspaceList } from '@/api/lightrag' // 🌟 引入新 API

import '@react-sigma/core/lib/style.css'
import '@react-sigma/graph-search/lib/style.css'

// Sigma 設定 (維持不變)
const createSigmaSettings = (isDarkTheme: boolean): Partial<SigmaSettings> => ({
  allowInvalidContainer: true,
  defaultNodeType: 'default',
  defaultEdgeType: 'curvedNoArrow',
  renderEdgeLabels: false,
  edgeProgramClasses: {
    arrow: EdgeArrowProgram,
    curvedArrow: EdgeCurvedArrowProgram,
    curvedNoArrow: createEdgeCurveProgram()
  },
  nodeProgramClasses: {
    default: NodeBorderProgram,
    circel: NodeCircleProgram,
    point: NodePointProgram
  },
  labelGridCellSize: 60,
  labelRenderedSizeThreshold: 12,
  enableEdgeEvents: true,
  labelColor: {
    color: isDarkTheme ? labelColorDarkTheme : labelColorLightTheme,
    attribute: 'labelColor'
  },
  edgeLabelColor: {
    color: isDarkTheme ? labelColorDarkTheme : labelColorLightTheme,
    attribute: 'labelColor'
  },
  edgeLabelSize: 8,
  labelSize: 12
})

const GraphEvents = () => {
  const registerEvents = useRegisterEvents()
  const sigma = useSigma()
  const [draggedNode, setDraggedNode] = useState<string | null>(null)

  useEffect(() => {
    registerEvents({
      downNode: (e) => {
        setDraggedNode(e.node)
        sigma.getGraph().setNodeAttribute(e.node, 'highlighted', true)
      },
      mousemovebody: (e) => {
        if (!draggedNode) return
        const pos = sigma.viewportToGraph(e)
        sigma.getGraph().setNodeAttribute(draggedNode, 'x', pos.x)
        sigma.getGraph().setNodeAttribute(draggedNode, 'y', pos.y)
        e.preventSigmaDefault()
        e.original.preventDefault()
        e.original.stopPropagation()
      },
      mouseup: () => {
        if (draggedNode) {
          setDraggedNode(null)
          sigma.getGraph().removeNodeAttribute(draggedNode, 'highlighted')
        }
      },
      mousedown: (e) => {
        const mouseEvent = e.original as MouseEvent;
        if (mouseEvent.buttons !== 0 && !sigma.getCustomBBox()) {
          sigma.setCustomBBox(sigma.getBBox())
        }
      }
    })
  }, [registerEvents, sigma, draggedNode])
  return null
}

const GraphViewer = () => {
  const [isThemeSwitching, setIsThemeSwitching] = useState(false)
  const [workspaces, setWorkspaces] = useState<{label: string, value: string}[]>([])
  // 🌟 修正 1：初始化時先從 localStorage 攞返之前揀咗嘅 Workspace
  const [currentWs, setCurrentWs] = useState<string>(() => {
    return localStorage.getItem('SELECTED_WORKSPACE') || ""
  })
  const sigmaRef = useRef<any>(null)
  const prevTheme = useRef<string>('')

  const selectedNode = useGraphStore.use.selectedNode()
  const focusedNode = useGraphStore.use.focusedNode()
  const moveToSelectedNode = useGraphStore.use.moveToSelectedNode()
  const isFetching = useGraphStore.use.isFetching()

  const showPropertyPanel = useSettingsStore.use.showPropertyPanel()
  const showNodeSearchBar = useSettingsStore.use.showNodeSearchBar()
  const enableNodeDrag = useSettingsStore.use.enableNodeDrag()
  const showLegend = useSettingsStore.use.showLegend()
  const theme = useSettingsStore.use.theme()

  // 1. 初始化攞 Workspace 清單
  useEffect(() => {
    getWorkspaceList().then(setWorkspaces).catch(console.error)
  }, [])
// 修改 src/features/GraphViewer.tsx
const handleWorkspaceChange = (value: string) => {
    setCurrentWs(value);
    localStorage.setItem('SELECTED_WORKSPACE', value);
    
    // 🌟 修正 3: 徹底清理 Store，防止舊數據殘留導致渲染錯誤
    const store = useGraphStore.getState();
    store.setSigmaInstance(null); // 銷毀舊畫布
    store.setSelectedNode(null);  // 清理選中點
    
    // 💡 重要：觸發 fetchGraphData 重新抓取
    // 喺你嘅 src/stores/graph.ts 入面應該有個 fetchData 嘅邏輯
    if (store.setGraphDataFetchAttempted) {
        store.setGraphDataFetchAttempted(false);
    }
    
    // 如果仲係有問題，最暴力但 100% 有效嘅方法係：
    window.location.reload(); 
};

  const memoizedSigmaSettings = useMemo(() => {
    const isDarkTheme = theme === 'dark'
    return createSigmaSettings(isDarkTheme)
  }, [theme])

  useEffect(() => {
    const isThemeChange = prevTheme.current && prevTheme.current !== theme
    if (isThemeChange) {
      setIsThemeSwitching(true)
      const timer = setTimeout(() => setIsThemeSwitching(false), 150)
      return () => clearTimeout(timer)
    }
    prevTheme.current = theme
  }, [theme])

  useEffect(() => {
    return () => {
      const sigma = useGraphStore.getState().sigmaInstance;
      if (sigma) {
        try {
          sigma.kill();
          useGraphStore.getState().setSigmaInstance(null);
        } catch (error) {
          console.error('Error cleaning up sigma instance:', error);
        }
      }
    };
  }, []);

  const onSearchFocus = useCallback((value: GraphSearchOption | null) => {
    if (value === null) useGraphStore.getState().setFocusedNode(null)
    else if (value.type === 'nodes') useGraphStore.getState().setFocusedNode(value.id)
  }, [])

  const onSearchSelect = useCallback((value: GraphSearchOption | null) => {
    if (value === null) useGraphStore.getState().setSelectedNode(null)
    else if (value.type === 'nodes') useGraphStore.getState().setSelectedNode(value.id, true)
  }, [])

  const autoFocusedNode = useMemo(() => focusedNode ?? selectedNode, [focusedNode, selectedNode])
  const searchInitSelectedNode = useMemo(
    (): OptionItem | null => (selectedNode ? { type: 'nodes', id: selectedNode } : null),
    [selectedNode]
  )

  return (
    <div className="relative h-full w-full overflow-hidden">
      {/* 🌟 頂部工具欄：加入 Workspace 選擇器 */}
      <div className="absolute top-2 left-2 z-50 flex items-center gap-2">
        <div className="flex items-center gap-2 bg-background/80 backdrop-blur-md p-1 px-3 rounded-lg border shadow-sm">
          <Database className="w-4 h-4 text-primary" />
          <Select value={currentWs} onValueChange={handleWorkspaceChange}>
            <SelectTrigger className="w-[200px] h-8 border-none bg-transparent focus:ring-0">
              <SelectValue placeholder="選擇數據空間..." />
            </SelectTrigger>
            <SelectContent>
              {workspaces.map((ws) => (
                <SelectItem key={ws.value} value={ws.value}>
                  {ws.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        
        <GraphLabels />
        
        {showNodeSearchBar && !isThemeSwitching && (
          <GraphSearch
            value={searchInitSelectedNode}
            onFocus={onSearchFocus}
            onChange={onSearchSelect}
          />
        )}
      </div>

      <SigmaContainer
        settings={memoizedSigmaSettings}
        className="!bg-background !size-full overflow-hidden"
        ref={sigmaRef}
      >
        <GraphControl />

        {enableNodeDrag && <GraphEvents />}

        <FocusOnNode node={autoFocusedNode} move={moveToSelectedNode} />

        <div className="bg-background/60 absolute bottom-2 left-2 flex flex-col rounded-xl border-2 backdrop-blur-lg">
          <LayoutsControl />
          <ZoomControl />
          <FullScreenControl />
          <LegendButton />
          <Settings />
        </div>

        {showPropertyPanel && (
          <div className="absolute top-2 right-2 z-10">
            <PropertiesView />
          </div>
        )}

        {showLegend && (
          <div className="absolute bottom-10 right-2 z-0">
            <Legend className="bg-background/60 backdrop-blur-lg" />
          </div>
        )}

        <SettingsDisplay />
      </SigmaContainer>

      {/* Loading overlay */}
      {(isFetching || isThemeSwitching) && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-50">
          <div className="text-center">
            <div className="mb-2 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto"></div>
            <p className="text-sm font-medium">
              {isThemeSwitching ? '正在切換主題...' : `正在加載 ${currentWs || '圖譜'} 數據...`}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default GraphViewer